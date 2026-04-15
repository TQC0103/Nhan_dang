import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner
from mmdet.core.sample_redistribution import (
    bins_to_hist,
    compute_face_sizes_from_boxes,
    get_bin_names,
    normalize_redistribution_cfg,
    assign_size_bins_from_sizes,
)


@BBOX_ASSIGNERS.register_module()
class ATSSAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (float): number of bbox selected in each level
    """

    def __init__(self,
                 topk,
                 mode=0,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 ignore_iof_thr=-1,
                 redistribution_cfg=None):
        self.topk = topk
        self.mode = mode
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.ignore_iof_thr = ignore_iof_thr
        self.redistribution_cfg = normalize_redistribution_cfg(redistribution_cfg)
        self.jsar_enabled = self.redistribution_cfg['ENABLE_JSAR']
        self.jsar_mode = self.redistribution_cfg['JSAR_MODE']
        self.jsar_topk = self.redistribution_cfg['JSAR_TOPK']
        self.jsar_center_radius_scale = self.redistribution_cfg['JSAR_CENTER_RADIUS_SCALE']
        self.jsar_min_pos_per_tiny_gt = self.redistribution_cfg['JSAR_MIN_POS_PER_TINY_GT']
        self.bin_edges = self.redistribution_cfg['ADAPTIVE_SR_BIN_EDGES']
        self.bin_names = get_bin_names(self.bin_edges)

    # https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py

    def _get_gt_size_info(self, gt_bboxes):
        gt_sizes = compute_face_sizes_from_boxes(gt_bboxes)
        gt_bins = assign_size_bins_from_sizes(gt_sizes, self.bin_edges)
        tiny_mask = gt_sizes < self.redistribution_cfg['JSAR_TINY_MAX_SIZE']
        small_mask = ((gt_sizes >= self.redistribution_cfg['JSAR_TINY_MAX_SIZE'])
                      & (gt_sizes < self.redistribution_cfg['JSAR_SMALL_MAX_SIZE']))
        return gt_sizes, gt_bins, tiny_mask, small_mask

    def _get_threshold_delta(self, tiny_mask, small_mask, device):
        deltas = torch.zeros((tiny_mask.shape[0], ), device=device)
        deltas[tiny_mask] = self.redistribution_cfg['JSAR_TINY_IOU_DELTA']
        deltas[small_mask] = self.redistribution_cfg['JSAR_SMALL_IOU_DELTA']
        return deltas

    def _get_center_mask(self, dist_min, gt_sizes, base_thr):
        center_mask = dist_min > base_thr
        if self.jsar_enabled and self.jsar_mode in ('size_aware_threshold', 'hybrid_fallback', 'soft_weight'):
            center_scale = torch.clamp(
                gt_sizes / max(self.redistribution_cfg['JSAR_TINY_MAX_SIZE'], 1.0),
                min=1.0,
                max=max(self.jsar_center_radius_scale, 1.0),
            )
            center_mask = dist_min * center_scale[None, :] > base_thr
        return center_mask

    def _expand_pos_for_small_faces(self,
                                    assigned_gt_inds,
                                    overlaps,
                                    distances,
                                    dist_min_full,
                                    gt_sizes,
                                    gt_bins):
        forced_mask = torch.zeros_like(assigned_gt_inds, dtype=torch.bool)
        soft_weights = torch.ones_like(overlaps[:, 0])
        if not self.jsar_enabled or self.jsar_mode not in ('hybrid_fallback', 'soft_weight'):
            return assigned_gt_inds, forced_mask, soft_weights

        temp = max(self.redistribution_cfg['JSAR_SOFT_WEIGHT_TEMPERATURE'], 1e-4)
        for gt_idx in range(gt_sizes.shape[0]):
            gt_size = gt_sizes[gt_idx]
            if gt_size >= self.redistribution_cfg['JSAR_SMALL_MAX_SIZE']:
                continue
            min_required = self.jsar_min_pos_per_tiny_gt
            if gt_size >= self.redistribution_cfg['JSAR_TINY_MAX_SIZE']:
                min_required = max(1, min_required - 1)
            current_pos = int((assigned_gt_inds == (gt_idx + 1)).sum().item())
            if current_pos >= min_required:
                continue
            candidate_mask = assigned_gt_inds == 0
            candidate_mask &= dist_min_full[:, gt_idx] > (-0.2 * self.jsar_center_radius_scale)
            if not candidate_mask.any():
                continue
            norm_dist = distances[:, gt_idx] / torch.clamp(gt_sizes[gt_idx], min=1.0)
            candidate_score = overlaps[:, gt_idx] - 0.05 * norm_dist
            candidate_score = candidate_score.masked_fill(~candidate_mask, float('-inf'))
            selectable = min(
                max(min_required - current_pos, 1),
                self.jsar_topk,
                int(candidate_mask.sum().item()),
            )
            if selectable <= 0:
                continue
            topk_scores, topk_indices = torch.topk(candidate_score, k=selectable, dim=0)
            valid_mask = torch.isfinite(topk_scores)
            if not valid_mask.any():
                continue
            topk_indices = topk_indices[valid_mask]
            assigned_gt_inds[topk_indices] = gt_idx + 1
            forced_mask[topk_indices] = True
            if self.jsar_mode == 'soft_weight':
                weights = torch.exp(-norm_dist[topk_indices] / temp) * torch.clamp(
                    overlaps[topk_indices, gt_idx] + 0.1,
                    min=0.05,
                    max=1.0,
                )
                soft_weights[topk_indices] = torch.clamp(weights, min=0.1, max=1.0)
        return assigned_gt_inds, forced_mask, soft_weights

    def assign(self,
               bboxes,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as postive
        6. limit the positive sample's center in gt


        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        INF = 100000000
        bboxes = bboxes[:, :4]
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)
        #print('AT1:', num_gt, num_bboxes)

        # compute iou between all bbox and gt
        overlaps = self.iou_calculator(bboxes, gt_bboxes)

        # assign 0 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # compute center distance between all bbox and gt
        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        gt_points = torch.stack((gt_cx, gt_cy), dim=1)

        gt_width = gt_bboxes[:,2] - gt_bboxes[:,0]
        gt_height = gt_bboxes[:,3] - gt_bboxes[:,1]
        gt_area = torch.sqrt( torch.clamp(gt_width*gt_height, min=1e-4) )
        gt_sizes, gt_bins, tiny_mask, small_mask = self._get_gt_size_info(gt_bboxes)

        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        bboxes_points = torch.stack((bboxes_cx, bboxes_cy), dim=1)

        distances = (bboxes_points[:, None, :] -
                     gt_points[None, :, :]).pow(2).sum(-1).sqrt()
        l_full = bboxes_cx[:, None] - gt_bboxes[:, 0][None, :]
        t_full = bboxes_cy[:, None] - gt_bboxes[:, 1][None, :]
        r_full = gt_bboxes[:, 2][None, :] - bboxes_cx[:, None]
        b_full = gt_bboxes[:, 3][None, :] - bboxes_cy[:, None]
        dist_min_full = torch.stack([l_full, t_full, r_full, b_full], dim=0).min(dim=0)[0]
        dist_min_full = dist_min_full / gt_area[None, :]

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            ignore_overlaps = self.iou_calculator(
                bboxes, gt_bboxes_ignore, mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            distances[ignore_idxs, :] = INF
            assigned_gt_inds[ignore_idxs] = -1

        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :] #(A,G)
            selectable_k = min(self.topk, bboxes_per_level)
            _, topk_idxs_per_level = distances_per_level.topk(
                selectable_k, dim=0, largest=False)
            #print('AT-LEVEL:', start_idx, end_idx, bboxes_per_level, topk_idxs_per_level.shape)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)# candidate anchors (topk*num_level_bboxes, G) = (AK, G)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)] #(AK,G)
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt
        if self.jsar_enabled and self.jsar_mode in ('size_aware_threshold', 'hybrid_fallback', 'soft_weight'):
            overlaps_thr_per_gt = torch.clamp(
                overlaps_thr_per_gt - self._get_threshold_delta(tiny_mask, small_mask, overlaps.device),
                min=0.0,
            )

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]
        #print('CAND:', candidate_idxs.shape, candidate_overlaps.shape, is_pos.shape)
        #print('BOXES:', bboxes_cx.shape)

        # limit the positive sample's center in gt
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        ep_bboxes_cx = bboxes_cx.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        ep_bboxes_cy = bboxes_cy.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        candidate_idxs = candidate_idxs.view(-1)

        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side
        l_ = ep_bboxes_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
        t_ = ep_bboxes_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_bboxes_cx[candidate_idxs].view(-1, num_gt)
        b_ = gt_bboxes[:, 3] - ep_bboxes_cy[candidate_idxs].view(-1, num_gt)
        #is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01
        dist_min = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] # (A,G)
        dist_min.div_(gt_area)
        #print('ATTT:', l_.shape, t_.shape, dist_min.shape, self.mode)
        if self.mode==0:
            center_thr = 0.001
        elif self.mode==1:
            center_thr = -0.25
        elif self.mode==2:
            center_thr = -0.15
            #dist_expand = torch.clamp(gt_area / 16.0, min=1.0, max=3.0)
            #dist_min.mul_(dist_expand)
            #is_in_gts = dist_min > -0.25
        elif self.mode==3:
            dist_expand = torch.clamp(gt_area / 16.0, min=1.0, max=6.0)
            dist_min.mul_(dist_expand)
            center_thr = -0.2
        elif self.mode==4:
            dist_expand = torch.clamp(gt_area / 16.0, min=0.5, max=6.0)
            dist_min.mul_(dist_expand)
            center_thr = -0.2
        elif self.mode==5:
            dist_div = torch.clamp(gt_area / 16.0, min=0.5, max=3.0)
            dist_min.div_(dist_div)
            center_thr = -0.2
        else:
            raise ValueError
        is_in_gts = dist_min > center_thr
        if self.jsar_enabled and self.jsar_mode in ('size_aware_threshold', 'hybrid_fallback', 'soft_weight'):
            is_in_gts = self._get_center_mask(dist_min, gt_sizes, center_thr)
        #print(gt_area.shape, is_in_gts.shape, is_pos.shape)
        is_pos = is_pos & is_in_gts

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1
        base_assigned_gt_inds = assigned_gt_inds.clone()
        base_pos_inds = torch.nonzero(base_assigned_gt_inds > 0, as_tuple=False).squeeze(-1)
        if base_pos_inds.numel() > 0:
            before_hist = bins_to_hist(
                gt_bins[base_assigned_gt_inds[base_pos_inds] - 1],
                len(self.bin_names),
            )
        else:
            before_hist = [0 for _ in range(len(self.bin_names))]
        forced_pos_mask = torch.zeros_like(assigned_gt_inds, dtype=torch.bool)
        soft_weights = torch.ones_like(max_overlaps)
        assigned_gt_inds, forced_pos_mask, soft_weights = self._expand_pos_for_small_faces(
            assigned_gt_inds,
            overlaps,
            distances,
            dist_min_full,
            gt_sizes,
            gt_bins,
        )
        if forced_pos_mask.any():
            forced_pos_inds = torch.nonzero(forced_pos_mask, as_tuple=False).squeeze(-1)
            forced_gt_inds = assigned_gt_inds[forced_pos_inds] - 1
            max_overlaps[forced_pos_inds] = overlaps[forced_pos_inds, forced_gt_inds]
        pos_inds_after = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze(-1)
        if pos_inds_after.numel() > 0:
            after_hist = bins_to_hist(
                gt_bins[assigned_gt_inds[pos_inds_after] - 1],
                len(self.bin_names),
            )
        else:
            after_hist = [0 for _ in range(len(self.bin_names))]

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        assign_result = AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
        assign_result.set_extra_property('jsar_before_hist', before_hist)
        assign_result.set_extra_property('jsar_after_hist', after_hist)
        assign_result.set_extra_property('gt_size_bins', gt_bins)
        assign_result.set_extra_property('jsar_forced_pos_mask', forced_pos_mask)
        assign_result.set_extra_property('jsar_soft_weights', soft_weights)
        return assign_result
