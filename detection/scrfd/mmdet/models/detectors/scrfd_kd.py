from copy import deepcopy
import warnings

import torch
import torch.nn.functional as F
from mmcv.runner import load_checkpoint

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_detector
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class SCRFDKD(SingleStageDetector):
    """SCRFD detector with output-level knowledge distillation.

    The implementation matches teacher/student outputs by stride so it can
    compare detectors with different neck depths. It also reduces anchor-wise
    outputs to a common representation, which makes cross-architecture
    distillation feasible when the teacher and student use different anchor
    layouts.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 teacher,
                 distill_cfg,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SCRFDKD, self).__init__(
            backbone, neck, bbox_head, train_cfg, test_cfg, pretrained)

        teacher_cfg = deepcopy(teacher)
        teacher_pretrained = teacher_cfg.pop('pretrained', None)
        self.teacher = build_detector(
            teacher_cfg, train_cfg=None, test_cfg=test_cfg)
        if teacher_pretrained:
            load_checkpoint(
                self.teacher,
                teacher_pretrained,
                map_location='cpu',
                strict=False)
        else:
            warnings.warn(
                'Teacher checkpoint is not set. Distillation from an '
                'uninitialized frozen teacher is usually not meaningful.',
                stacklevel=2)

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.distill_cfg = distill_cfg
        self.cls_weight = distill_cfg.get('cls_weight', 0.5)
        self.bbox_weight = distill_cfg.get('bbox_weight', 0.5)
        self.temperature = distill_cfg.get('temperature', 4.0)
        self.cls_loss_type = distill_cfg.get('cls_loss_type', 'bce')
        self.match_by_stride = distill_cfg.get('match_by_stride', True)
        self.anchor_reduce = distill_cfg.get('anchor_reduce', 'mean')

    def train(self, mode=True):
        super(SCRFDKD, self).train(mode)
        self.teacher.eval()
        return self

    def _stride_to_int(self, stride):
        if isinstance(stride, (tuple, list)):
            return int(stride[0])
        return int(stride)

    def _get_head_strides(self, head, num_levels):
        strides = getattr(getattr(head, 'anchor_generator', None), 'strides', None)
        if not strides:
            return [None] * num_levels
        return [self._stride_to_int(stride) for stride in strides[:num_levels]]

    def _get_level_pairs(self, student_levels, teacher_levels):
        if not self.match_by_stride:
            return [(i, i) for i in range(min(len(student_levels), len(teacher_levels)))]

        student_strides = self._get_head_strides(self.bbox_head, len(student_levels))
        teacher_strides = self._get_head_strides(
            self.teacher.bbox_head, len(teacher_levels))

        student_map = {
            stride: idx for idx, stride in enumerate(student_strides)
            if stride is not None
        }
        teacher_map = {
            stride: idx for idx, stride in enumerate(teacher_strides)
            if stride is not None
        }
        common_strides = [stride for stride in student_strides if stride in teacher_map]
        if common_strides:
            return [(student_map[stride], teacher_map[stride]) for stride in common_strides]

        return [(i, i) for i in range(min(len(student_levels), len(teacher_levels)))]

    def _resize_like(self, src, ref):
        if src.shape[-2:] == ref.shape[-2:]:
            return src
        return F.interpolate(
            src,
            size=ref.shape[-2:],
            mode='bilinear',
            align_corners=False)

    def _reduce_anchor_dim(self, pred, per_anchor_channels):
        if pred.shape[1] == per_anchor_channels:
            return pred
        if pred.shape[1] % per_anchor_channels != 0:
            return pred.mean(dim=1, keepdim=True)

        num_anchors = pred.shape[1] // per_anchor_channels
        pred = pred.reshape(
            pred.shape[0],
            num_anchors,
            per_anchor_channels,
            pred.shape[2],
            pred.shape[3])

        if self.anchor_reduce == 'sum':
            return pred.sum(dim=1)
        return pred.mean(dim=1)

    def _align_cls_level(self, pred, head):
        return self._reduce_anchor_dim(pred, head.cls_out_channels)

    def _align_bbox_level(self, pred, head):
        per_anchor_channels = 4 * (head.reg_max + 1) if head.use_dfl else 4
        return self._reduce_anchor_dim(pred, per_anchor_channels)

    def _classification_distill_loss(self, student_pred, teacher_pred):
        temperature = self.temperature
        if self.cls_loss_type == 'mse':
            return F.mse_loss(
                torch.sigmoid(student_pred / temperature),
                torch.sigmoid(teacher_pred / temperature)) * (temperature ** 2)

        return F.binary_cross_entropy_with_logits(
            student_pred / temperature,
            torch.sigmoid(teacher_pred / temperature)) * (temperature ** 2)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_keypointss=None,
                      gt_bboxes_ignore=None):
        super(SingleStageDetector, self).forward_train(img, img_metas)

        x = self.extract_feat(img)
        student_outs = self.bbox_head(x)

        self.teacher.eval()
        with torch.no_grad():
            teacher_outs = self.teacher.bbox_head(self.teacher.extract_feat(img))

        task_losses = self.bbox_head.forward_train(
            x,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_keypointss,
            gt_bboxes_ignore)
        distill_losses = self.compute_distill_loss(student_outs, teacher_outs)

        losses = {}
        losses.update(task_losses)
        losses.update(distill_losses)
        return losses

    def compute_distill_loss(self, student_outs, teacher_outs):
        student_cls, student_bbox, _ = student_outs
        teacher_cls, teacher_bbox, _ = teacher_outs

        level_pairs = self._get_level_pairs(student_cls, teacher_cls)
        if not level_pairs:
            zero = student_cls[0].sum() * 0
            return {
                'loss_cls_distill': zero,
                'loss_bbox_distill': zero,
                'loss_distill': zero,
            }

        cls_loss = 0
        bbox_loss = 0
        for student_idx, teacher_idx in level_pairs:
            s_cls = self._align_cls_level(student_cls[student_idx], self.bbox_head)
            t_cls = self._align_cls_level(
                teacher_cls[teacher_idx], self.teacher.bbox_head)
            t_cls = self._resize_like(t_cls, s_cls)
            cls_loss = cls_loss + self._classification_distill_loss(s_cls, t_cls)

            s_bbox = self._align_bbox_level(
                student_bbox[student_idx], self.bbox_head)
            t_bbox = self._align_bbox_level(
                teacher_bbox[teacher_idx], self.teacher.bbox_head)
            t_bbox = self._resize_like(t_bbox, s_bbox)
            if s_bbox.shape[1] != t_bbox.shape[1]:
                raise ValueError(
                    'Student and teacher bbox outputs are incompatible after '
                    'anchor reduction. Please use the same regression format '
                    f'or adapt distillation. Student={s_bbox.shape}, '
                    f'teacher={t_bbox.shape}')
            bbox_loss = bbox_loss + F.mse_loss(s_bbox, t_bbox)

        cls_loss = cls_loss / len(level_pairs) * self.cls_weight
        bbox_loss = bbox_loss / len(level_pairs) * self.bbox_weight

        return {
            'loss_cls_distill': cls_loss,
            'loss_bbox_distill': bbox_loss,
            'loss_distill': cls_loss + bbox_loss
        }

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results
