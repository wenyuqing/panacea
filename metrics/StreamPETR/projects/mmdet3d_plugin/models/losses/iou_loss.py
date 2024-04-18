import torch
from torch import nn

from mmdet.models.builder import LOSSES

@LOSSES.register_module()
class IOULoss(nn.Module):
    def __init__(self, loss_type="iou", return_iou=True, reduction="mean"):
        super(IOULoss, self).__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        self.return_iou = return_iou

    def forward(self, pred, target, weight=None, avg_factor=1.0):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(
            pred_right, target_right
        )
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(
            pred_right, target_right
        )

        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(
            pred_top, target_top
        )
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(
            pred_top, target_top
        )

        ac_uion = g_w_intersect * g_h_intersect + 1e-7
        area_intersect = w_intersect * h_intersect

        area_union = target_area + pred_area - area_intersect
        ious = area_intersect / (area_union + 1e-7)
        gious = ious - (ac_uion - area_union) / ac_uion

        if self.loss_type == "iou":
            losses = -torch.log(ious)
        elif self.loss_type == "linear_iou":
            losses = 1 - ious
        elif self.loss_type == "giou":
            losses = 1 - gious
        else:
            raise NotImplementedError

        if self.reduction == "mean":
            if avg_factor is None:
                losses = losses.mean()
            else:
                losses = losses.sum() / avg_factor
        
        elif self.reduction == "sum":
            losses = losses.sum()

        if self.return_iou:
            return losses, ious
        else:
            return losses