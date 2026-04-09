import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDistillLoss(nn.Module):
    """KL Divergence loss for classification score distillation.

    Distills the soft predictions from teacher to student using KL divergence
    with temperature scaling.

    Args:
        temperature (float): Temperature for softening probability distributions.
            Higher values produce softer distributions. Default: 4.0
        reduction (str): Specifies the reduction to apply to the output.
            Can be 'none', 'mean', 'batchmean', or 'sum'. Default: 'batchmean'
    """

    def __init__(self, temperature=4.0, reduction='batchmean'):
        super(KLDistillLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, s_scores, t_scores):
        """Compute KL divergence between student and teacher scores.

        Args:
            s_scores (list[Tensor]): Student classification scores, each is
                a 4D tensor of shape (N, num_classes, H, W)
            t_scores (list[Tensor]): Teacher classification scores, same shape
                as student scores

        Returns:
            Tensor: KL divergence loss
        """
        loss = 0
        count = 0
        for s, t in zip(s_scores, t_scores):
            s_soft = F.log_softmax(s / self.temperature, dim=1)
            t_soft = F.softmax(t / self.temperature, dim=1)
            loss += F.kl_div(s_soft, t_soft, reduction=self.reduction) * (
                self.temperature ** 2)
            count += 1
        return loss / max(count, 1)


class L2DistillLoss(nn.Module):
    """L2 loss for bbox prediction distillation.

    Distills the bounding box predictions from teacher to student using
    mean squared error loss.

    Args:
        reduction (str): Specifies the reduction to apply to the output.
            Can be 'none', 'mean', or 'sum'. Default: 'mean'
    """

    def __init__(self, reduction='mean'):
        super(L2DistillLoss, self).__init__()
        self.reduction = reduction

    def forward(self, s_bbox, t_bbox):
        """Compute L2 loss between student and teacher bbox predictions.

        Args:
            s_bbox (list[Tensor]): Student bbox predictions, each is
                a tensor of shape (N, 4*(reg_max+1), H, W)
            t_bbox (list[Tensor]): Teacher bbox predictions, same shape
                as student predictions

        Returns:
            Tensor: L2 distillation loss
        """
        loss = 0
        count = 0
        for s, t in zip(s_bbox, t_bbox):
            # Handle shape mismatch if teacher and student have different strides
            if s.shape != t.shape:
                # Resize teacher output to match student
                s_size = (s.shape[2], s.shape[3])
                t = F.interpolate(t, size=s_size, mode='bilinear', align_corners=False)
            loss += F.mse_loss(s, t, reduction=self.reduction)
            count += 1
        return loss / max(count, 1)


class CombinedDistillLoss(nn.Module):
    """Combined distillation loss for SCRFD.

    Combines KL divergence loss on classification scores and
    L2 loss on bbox predictions.

    Args:
        cls_weight (float): Weight for classification distillation loss. Default: 0.5
        bbox_weight (float): Weight for bbox distillation loss. Default: 0.5
        temperature (float): Temperature for KL divergence. Default: 4.0
    """

    def __init__(self, cls_weight=0.5, bbox_weight=0.5, temperature=4.0):
        super(CombinedDistillLoss, self).__init__()
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.kd_loss = KLDistillLoss(temperature=temperature)
        self.l2_loss = L2DistillLoss()

    def forward(self, s_outs, t_outs):
        """Compute combined distillation loss.

        Args:
            s_outs (tuple): Student outputs (cls_scores, bbox_preds, kps_preds)
            t_outs (tuple): Teacher outputs (cls_scores, bbox_preds, kps_preds)

        Returns:
            dict: Dictionary containing:
                - loss_cls_distill: Classification distillation loss
                - loss_bbox_distill: Bbox distillation loss
                - loss_distill: Combined distillation loss
        """
        s_cls, s_bbox, _ = s_outs
        t_cls, t_bbox, _ = t_outs

        cls_loss = self.kd_loss(s_cls, t_cls) * self.cls_weight
        bbox_loss = self.l2_loss(s_bbox, t_bbox) * self.bbox_weight

        return {
            'loss_cls_distill': cls_loss,
            'loss_bbox_distill': bbox_loss,
            'loss_distill': cls_loss + bbox_loss
        }
