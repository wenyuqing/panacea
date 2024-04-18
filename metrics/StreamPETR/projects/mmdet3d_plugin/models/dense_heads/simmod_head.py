import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import force_fp32

from mmdet.core import multi_apply, multi_apply, reduce_mean
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox, denormalize_bbox
from mmdet3d.core.bbox.structures.utils import rotation_3d_in_axis
from mmdet3d.core.bbox import BboxOverlaps3D

import pdb

@HEADS.register_module()
class SimMODHead(DETRHead):
    def __init__(
        self,
        *args,
        with_box_refine=True,
        transformer=None,
        bbox_coder=None,
        num_cls_fcs=2,
        input_proposal_channel=None,
        num_input_proj=2,
        input_pos_channel=3,
        combine_scores=False,
        detach_proposal_positions=False,
        using_pos_embeddings=True,
        proposal_level_embeddings=False,
        proposal_cam_embeddings=False,
        proposal_embeddings_additive=True,
        only_use_xyz_pos=True,
        use_level_embeddings=False,
        use_cam_embeddings=False,
        use_cam_level_embeddings=False,
        code_weights=None,
        num_camera=6,
        num_level=4,
        compute_loss_iou3d=False,
        loss_iou3d_weight=1.0,
        using_queries=False,
        using_guided_assign=False,
        **kwargs,
    ):
        self.with_box_refine = with_box_refine
        self.code_size = kwargs.get("code_size", 10)
        self.code_weights = code_weights if code_weights is not None else [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.num_cls_fcs = num_cls_fcs - 1

        self.num_input_proj = num_input_proj
        self.input_proposal_channel = input_proposal_channel
        # detach the positions of proposals so that two-stage can learn the residual value
        self.detach_proposal_positions = detach_proposal_positions
        # using positional embeddings
        self.using_pos_embeddings = using_pos_embeddings
        # learn positional embeddings from positions, view, fpn_level
        self.input_pos_channel = input_pos_channel
        self.only_use_xyz_pos = only_use_xyz_pos

        self.proposal_level_embeddings = proposal_level_embeddings
        self.proposal_cam_embeddings = proposal_cam_embeddings
        self.proposal_embeddings_additive = proposal_embeddings_additive

        # combine the classification scores of two stages
        self.combine_scores = combine_scores

        # mlvl_feats with level & cam embeddings
        self.use_cam_embeddings = use_cam_embeddings
        self.use_level_embeddings = use_level_embeddings
        self.use_cam_level_embeddings = use_cam_level_embeddings
        
        # using parameterized queries instead of proposals, for ablation only
        self.using_queries = using_queries

        self.using_guided_assign = using_guided_assign

        self.num_camera = num_camera
        self.num_level = num_level

        super(SimMODHead, self).__init__(
            *args, transformer=transformer, **kwargs
        )

        self.iou3d_calculator = BboxOverlaps3D(coordinate='lidar')
        self.compute_loss_iou3d = compute_loss_iou3d
        if self.compute_loss_iou3d:
            self.iou3d_criterion = nn.BCEWithLogitsLoss(weight=torch.tensor(loss_iou3d_weight))

        self.code_weights = nn.Parameter(
            torch.tensor(self.code_weights, requires_grad=False), requires_grad=False
        )
        
        self.pc_range = nn.Parameter(torch.tensor(
            self.bbox_coder.pc_range), requires_grad=False)

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU(inplace=True))
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = self.transformer.decoder.num_layers
        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])
        
        # init level & camera embeddings
        if self.use_cam_embeddings:
            self.cams_embeds = nn.Embedding(self.num_camera, self.embed_dims)
            
        if self.use_level_embeddings:
            self.level_embeds = nn.Embedding(self.num_level, self.embed_dims)
        
        if self.use_cam_level_embeddings:
            self.cam_level_embeddings = nn.Embedding(self.num_camera * self.num_level, self.embed_dims)

        if self.proposal_embeddings_additive:
            if self.proposal_level_embeddings:
                self.proposal_level_embeds = nn.Embedding(self.num_camera, self.embed_dims)
            
            if self.proposal_cam_embeddings:
                self.proposal_cam_embeds = nn.Embedding(self.num_camera, self.embed_dims)
        
        else:
            if self.proposal_level_embeddings:
                self.proposal_level_embeds = nn.Embedding(self.num_camera, self.embed_dims // 2)
            
            if self.proposal_cam_embeddings:
                self.proposal_cam_embeds = nn.Embedding(self.num_camera, self.embed_dims // 2)

        input_proj_layers = []
        in_channels = self.input_proposal_channel
        for _ in range(self.num_input_proj):
            input_proj_layers.extend(
                [
                    Linear(in_channels, self.embed_dims),
                    nn.LayerNorm(self.embed_dims),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = self.embed_dims
        
        self.input_proj = nn.Sequential(*input_proj_layers)

        if self.using_pos_embeddings:
            self.pos_proj = Linear(self.input_pos_channel, self.embed_dims)

            if not self.proposal_embeddings_additive and (self.proposal_level_embeddings or self.proposal_cam_embeddings):
                in_pos_channels = self.embed_dims
                if self.proposal_level_embeddings:
                    in_pos_channels += self.embed_dims // 2
                if self.proposal_cam_embeddings:
                    in_pos_channels += self.embed_dims // 2
                
                self.pos_reduce = Linear(in_pos_channels, self.embed_dims)
        
        if self.using_queries:
            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
            self.reference_pts_fc = nn.Linear(self.embed_dims, 3)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        
        if self.use_cam_embeddings:
            nn.init.normal_(self.cams_embeds.weight, std=0.1)
        
        if self.use_level_embeddings:
            nn.init.normal_(self.level_embeds.weight, std=0.1)

        if self.use_cam_level_embeddings:
            nn.init.normal_(self.cam_level_embeddings.weight, std=0.1)

    def normalize_positions(self, proposal_positions):
        normalized_proposal_positions = proposal_positions.clone()
        num_pos_dim = 3
        for k in range(num_pos_dim):
            normalized_proposal_positions[..., k] = (
                normalized_proposal_positions[..., k] - self.pc_range[k]
            ) / (self.pc_range[k + num_pos_dim] - self.pc_range[k])

        return normalized_proposal_positions
    
    def get_pos_embeddings(self, proposal_positions):
        proposal_xyz, proposal_view, proposal_level = torch.split(proposal_positions, 
            [3, 1, 1], dim=2)
        
        batch, num_proposal = proposal_view.shape[:2]

        if self.only_use_xyz_pos:
            query_pos = self.pos_proj(proposal_xyz)
        else:
            query_pos = self.pos_proj(proposal_positions[..., :self.input_pos_channel])

        if self.proposal_cam_embeddings:
            query_pos_cam = self.proposal_cam_embeds.weight[proposal_view.view(-1).long()]
            query_pos_cam = query_pos_cam.view(batch, num_proposal, -1)

            if self.proposal_embeddings_additive:
                query_pos = query_pos + query_pos_cam
            else:
                query_pos = torch.cat((query_pos, query_pos_cam), dim=2)

        if self.proposal_level_embeddings:
            query_pos_level = self.proposal_level_embeds.weight[proposal_level.view(-1).long()]
            query_pos_level = query_pos_level.view(batch, num_proposal, -1)

            if self.proposal_embeddings_additive:
                query_pos = query_pos + query_pos_level
            else:
                query_pos = torch.cat((query_pos, query_pos_level), dim=2)
        
        # reduce channels
        if not self.proposal_embeddings_additive and (self.proposal_level_embeddings or self.proposal_cam_embeddings):
            query_pos = self.pos_reduce(query_pos)

        return query_pos

    def forward(
        self,
        mlvl_feats,
        img_metas,
        proposal_features,
        proposal_positions,
        proposal_scores=None,
        valid_ranges=None,
        raw_img=None,
    ):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        if self.using_queries:
            query_embeds = self.query_embedding.weight
            query_embeds, query_pos = torch.split(query_embeds, [self.embed_dims, self.embed_dims], dim=1)
            bs = mlvl_feats[0].shape[0]

            query_embeds = query_embeds.unsqueeze(0).repeat(bs, 1, 1)
            query_pos = query_pos.unsqueeze(0).repeat(bs, 1, 1)

            if self.reference_pts_fc:
                normalized_proposal_positions = self.reference_pts_fc(query_pos)

        else:
            # map cls+reg features to input_features
            query_embeds = self.input_proj(proposal_features)

            if self.detach_proposal_positions:
                proposal_positions = proposal_positions.detach()

            normalized_proposal_positions = self.normalize_positions(proposal_positions)
            if self.using_pos_embeddings:
                query_pos = self.get_pos_embeddings(normalized_proposal_positions)
            else:
                query_pos = None        
        
        input_mlvl_feats = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape

            if self.use_cam_embeddings:
                feat = feat + self.cams_embeds.weight[None, ..., None, None]
            
            if self.use_level_embeddings:
                level_embed = self.level_embeds.weight[lvl]
                feat = feat + level_embed.view(1, 1, -1, 1, 1)
            
            if self.use_cam_level_embeddings:
                cam_level_embed = self.cam_level_embeddings.weight.view(self.num_camera, self.num_level, self.embed_dims)
                cam_level_embed = cam_level_embed[:, lvl]
                # [num_cam, c] ==> [b, num_cam, c, h, w]
                feat = feat + cam_level_embed[None, ..., None, None]
            
            input_mlvl_feats.append(feat)

        hs, init_reference, inter_references, inter_outputs = self.transformer(
            input_mlvl_feats,
            query=query_embeds,
            query_pos=query_pos,
            query_reference_points=normalized_proposal_positions[..., :3],
            reg_branches=self.reg_branches,
            img_metas=img_metas,
            valid_ranges=valid_ranges,
            raw_img=raw_img,
            proposal_views=proposal_positions[..., 3],
            proposal_levels=proposal_positions[..., 4],
        )

        # [num_dec, N, B, C] ==> [num_dec, B, N, C]
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]

            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = inter_outputs[lvl]

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3

            # tmp[..., 0:2] += reference[..., 0:2]
            # tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            # tmp[..., 4:5] += reference[..., 2:3]
            # tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            # tmp[..., 0:1] = (
            #     tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            # )
            # tmp[..., 1:2] = (
            #     tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            # )
            # tmp[..., 4:5] = (
            #     tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            # )

            tmp[..., 0:3] += reference[..., 0:3]
            tmp[..., 0:3] = tmp[..., 0:3].sigmoid()
            tmp[..., 0:3] = (tmp[..., 0:3] * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3])

            outputs_classes.append(outputs_class)
            outputs_coords.append(tmp)
        
        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        if self.combine_scores:
            # combine the scores of two stages
            outputs_classes_sig = outputs_classes.sigmoid()
            outputs_classes_combined = outputs_classes_sig * proposal_scores[None, ..., None].detach()
            outputs_classes = inverse_sigmoid(outputs_classes_combined)

        outs = {
            "all_cls_scores": outputs_classes,
            "all_bbox_preds": outputs_coords,
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
        }

        return outs

    def _get_target_single(
        self, cls_score, bbox_pred, gt_labels, gt_bboxes, 
        gt_bboxes_ignore=None, preds_match_idxs=None,
    ):
        """ "Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        num_gt = gt_bboxes.shape[0]

        if self.using_guided_assign:
            '''Hungarian assignment with pre-defined matching'''
            assign_result = self.assigner.restricted_assign(
                bbox_pred, cls_score, gt_bboxes, gt_labels, 
                gt_bboxes_ignore, pre_assigns=preds_match_idxs,
            )
            sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
            pos_inds = sampling_result.pos_inds
            neg_inds = sampling_result.neg_inds

            ''' Debug and Anaylze '''
            # naive_assign_result = self.assigner.assign(
            #     bbox_pred, cls_score, gt_bboxes, gt_labels, gt_bboxes_ignore
            # )
            # naive_sampling_result = self.sampler.sample(naive_assign_result, bbox_pred, gt_bboxes)
            # naive_assigns = gt_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
            # naive_assigns[naive_sampling_result.pos_inds] = naive_sampling_result.pos_assigned_gt_inds
            # naive_consist = (naive_assigns == preds_match_idxs).sum().item()
            
            # guided_assigns = gt_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
            # guided_assigns[sampling_result.pos_inds] = sampling_result.pos_assigned_gt_inds
            # guided_consist = (guided_assigns == preds_match_idxs).sum().item()
            
            # if guided_consist > naive_consist:
            #     print('Naive, Guided, Total: ', naive_consist, guided_consist, preds_match_idxs.numel())
        
        else:
            # assigner and sampler
            assign_result = self.assigner.assign(
                bbox_pred, cls_score, gt_bboxes, gt_labels, gt_bboxes_ignore
            )
            sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
            pos_inds = sampling_result.pos_inds
            neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :9]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        if num_gt > 0:
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)

    def get_targets(
        self,
        cls_scores_list,
        bbox_preds_list,
        gt_bboxes_list,
        gt_labels_list,
        gt_bboxes_ignore_list=None,
        preds_match_idxs_list=None,
    ):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert (
            gt_bboxes_ignore_list is None
        ), "Only supports for gt_bboxes_ignore setting to None."
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single,
            cls_scores_list,
            bbox_preds_list,
            gt_labels_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            preds_match_idxs_list,
        )

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        
        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def loss_single(
        self,
        cls_scores,
        bbox_preds,
        gt_bboxes_list,
        gt_labels_list,
        gt_bboxes_ignore_list=None,
        preds_match_idxs=None,
    ):
        """ "Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        preds_match_idxs_list = [preds_match_idxs[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(
            cls_scores_list,
            bbox_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            gt_bboxes_ignore_list,
            preds_match_idxs_list,
        )
        
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets
        
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor
        )

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights[:, :10] * self.code_weights

        '''
        bbox_targets (tensor): batch x 9, 
        include gravity_center (3) + sizes (3) + yaw (1) + velocity (2)
        '''

        # computing IoU loss for positive samples
        if self.compute_loss_iou3d:            
            pos_bbox_preds_norm = bbox_preds[isnotnan]
            pos_bbox_preds = denormalize_bbox(pos_bbox_preds_norm[:, :10], self.pc_range)
            
            with torch.no_grad():
                target_ious = self.compute_bboxes_ious(pos_bbox_preds, bbox_targets[isnotnan])
                target_ious = torch.clamp(2 * target_ious - 0.5, min=0.0, max=1.0)
            
            pred_ious = pos_bbox_preds_norm[:, -1]
            loss_iou = self.iou3d_criterion(pred_ious, target_ious)
            loss_iou = torch.nan_to_num(loss_iou)
        else:
            loss_iou = None

        if isnotnan.sum() > 0:
            loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10],
                bbox_weights[isnotnan, :10], avg_factor=num_total_pos,
            )
        else:
            loss_bbox = self.loss_bbox(
                bbox_preds[:, :10], bbox_preds.new_zeros(bbox_preds.shape[0], 10),
                bbox_weights[:, :10], avg_factor=bbox_preds.shape[0],
            )
            loss_bbox = loss_bbox * 0.0

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        
        return loss_cls, loss_bbox, loss_iou

    @force_fp32(apply_to=("preds_dicts"))
    def loss(self, gt_bboxes_list, gt_labels_list, preds_dicts, 
            gt_bboxes_ignore=None, preds_match_idxs=None):
        """ "Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, (
            f"{self.__class__.__name__} only supports "
            f"for gt_bboxes_ignore setting to None."
        )

        all_cls_scores = preds_dicts["all_cls_scores"]
        all_bbox_preds = preds_dicts["all_bbox_preds"]
        enc_cls_scores = preds_dicts["enc_cls_scores"]
        enc_bbox_preds = preds_dicts["enc_bbox_preds"]

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [
            torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1).to(
                device
            )
            for gt_bboxes in gt_bboxes_list
        ]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]
        all_preds_match_idxs = [preds_match_idxs for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_bbox_preds,
            all_gt_bboxes_list,
            all_gt_labels_list,
            all_gt_bboxes_ignore_list,
            all_preds_match_idxs,
        )

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = self.loss_single(
                enc_cls_scores,
                enc_bbox_preds,
                gt_bboxes_list,
                binary_labels_list,
                gt_bboxes_ignore,
            )
            loss_dict["enc_loss_cls"] = enc_loss_cls
            loss_dict["enc_loss_bbox"] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict["loss_cls"] = losses_cls[-1]
        loss_dict["loss_bbox"] = losses_bbox[-1]

        if self.compute_loss_iou3d:
            loss_dict["loss_iou"] = losses_iou[-1]
        
        for num_dec_layer in range(len(losses_cls) - 1):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = losses_cls[num_dec_layer]
            loss_dict[f"d{num_dec_layer}.loss_bbox"] = losses_bbox[num_dec_layer]

            if self.compute_loss_iou3d:
                loss_dict[f"d{num_dec_layer}.loss_iou"] = losses_iou[num_dec_layer]
        
        return loss_dict
    
    def compute_bboxes_ious(self, preds, targets):
        preds[:, 2] = preds[:, 2] - 0.5 * preds[:, 5]
        targets[:, 2] = targets[:, 2] - 0.5 * targets[:, 5]
        iou_3d = self.iou3d_calculator(preds, targets)

        return torch.diag(iou_3d)

    @force_fp32(apply_to=("preds_dicts"))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        preds_dicts = self.bbox_coder.decode(preds_dicts, with_ious=self.compute_loss_iou3d)
        num_samples = len(preds_dicts)
        ret_list = []
        
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds["bboxes"]
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]["box_type_3d"](bboxes, 9)
            scores = preds["scores"]
            labels = preds["labels"]
            ret_list.append([bboxes, scores, labels])

        return ret_list
        