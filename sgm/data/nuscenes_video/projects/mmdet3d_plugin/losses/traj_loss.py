#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from mmdet.models import LOSSES
from projects.mmdet3d_plugin.models.utils.misc import topk_gather

@LOSSES.register_module()
class TrajLossv1(nn.Module):
    """
    MTP loss modified to include variances. Uses MSE for mode selection.
    Can also be used with
    Multipath outputs, with residuals added to anchors.
    """

    def __init__(self, cls_loss_weight=1., ade_loss_weight=1., fde_loss_weight=0.25):
        """
        Initialize MTP loss
        :param args: Dictionary with the following (optional) keys
            use_variance: bool, whether or not to use variances for computing
            regression component of loss,
                default: False
            alpha: float, relative weight assigned to classification component,
            compared to regression component
                of loss, default: 1
        """
        super(TrajLossv1, self).__init__()
        self.cls_loss_weight = cls_loss_weight
        self.ade_loss_weight = ade_loss_weight
        self.fde_loss_weight = fde_loss_weight

    def forward(self,
                traj_prob, # B, Q, M
                traj_preds, # B, Q, M, S, 4
                gt_future_traj, # B*Q, M, S, 2
                gt_future_traj_valid_mask,# B*Q, M, S, 2
                avg_factor): 
        """
        Compute MTP loss
        :param predictions: Dictionary with 'traj': predicted trajectories
        and 'probs': mode (log) probabilities
        :param ground_truth: Either a tensor with ground truth trajectories
        or a dictionary
        :return:
        """
        # Unpack arguments
        if avg_factor is None:
            avg_factor = gt_future_traj.shape[0]
        traj_preds = traj_preds.flatten(0, 1)
        traj_prob = traj_prob.flatten(0, 1)
        traj_prob = F.log_softmax(traj_prob, dim=-1)
        gt_future_traj_valid_mask = gt_future_traj_valid_mask[..., 0]
        valid_steps = gt_future_traj_valid_mask.sum(dim=-1) # B*Q, M
        final_inds = (valid_steps - 1).clamp(0).long().unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 2)
        pred_final = torch.gather(traj_preds, 2, final_inds)
        gt_final = torch.gather(gt_future_traj, 2, final_inds)
        
        min_fde_norm = (torch.norm(pred_final - gt_final, p=2, dim=-1) * gt_future_traj_valid_mask).sum(dim=-1)
        _, fde_mode = torch.topk(-min_fde_norm, 1, dim=1) 

        cls_mask = valid_steps > 0
        l2_norm = (torch.norm(traj_preds[:, :, :, : 2] - gt_future_traj, p=2, dim=-1) * gt_future_traj_valid_mask).sum(dim=-1)  # B*Q, M
        _, best_mode = torch.topk(-l2_norm, 1, dim=1)
        fde_best = topk_gather(pred_final, fde_mode)
        pred_best = topk_gather(traj_preds, best_mode)
        cls_best = topk_gather(traj_prob, best_mode)


        reg_mask = gt_future_traj_valid_mask[:, :1].unsqueeze(-1)
        gt_reg = gt_future_traj[:,:1]
        l_reg = torch.abs(gt_reg - pred_best)
        l_reg = l_reg * reg_mask

        l_min_fde = torch.abs(gt_final[:, 0:1] - fde_best)
        l_min_fde = l_min_fde * reg_mask[:, :, 0:1]

        l_class = -cls_best
        l_class = l_class * cls_mask[..., 0:1]
        l_reg = torch.sum(l_reg)/(avg_factor * 12)
        l_class = torch.sum(l_class)/avg_factor
        l_min_fde = torch.sum(l_min_fde)/avg_factor

        loss = l_class * self.cls_loss_weight + l_reg * self.ade_loss_weight + l_min_fde * self.fde_loss_weight
        return loss
