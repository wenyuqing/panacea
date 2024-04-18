import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

from collections import defaultdict
from abc import abstractmethod
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init, Scale
from mmcv.runner import force_fp32

from mmdet.core import multi_apply, reduce_mean
from mmdet3d.core import box3d_multiclass_nms, limit_period, xywhr2xyxyr
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.builder import HEADS, build_loss
from mmdet3d.models.dense_heads import BaseMono3DDenseHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmdet3d.core import (
    CameraInstance3DBoxes,
    bbox3d2result,
)
from mmcv.parallel import DataContainer as DC
from ..utils.misc import multi_apply_dic
import pdb

INF = 1e8


@HEADS.register_module()
class FCOSMono3D_ProposalHead(BaseMono3DDenseHead):
    """Anchor-free head used in FCOS3D.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: True.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: True.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: True.
        centerness_alpha: Parameter used to adjust the intensity attenuation
            from the center to the periphery. Default: 2.5.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_dir (dict): Config of direction classification loss.
        loss_attr (dict): Config of attribute classification loss.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        centerness_branch (tuple[int]): Channels for centerness branch.
            Default: (64, ).
    """  # noqa: E501

    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels=256,
        stacked_convs=4,
        strides=(4, 8, 16, 32, 64),
        dcn_on_last_conv=False,
        conv_bias="auto",
        background_label=None,
        use_direction_classifier=True,
        diff_rad_by_sin=True,
        dir_offset=0,
        regress_ranges=((-1, 48), (48, 96), (96, 192), (192, 1e4)),
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        using_2d_centers=False,
        # proposals
        num_proposal=600,
        generate_multi_class_proposals=False,
        generate_proposal_per_image=True,
        objectness_with_centerness=True,
        objectness_max_pooling=True,
        random_objectness_with_teacher=0.0,
        random_proposal_drop=False,
        random_proposal_drop_upper_bound=1.0,
        random_proposal_drop_lower_bound=0.7,
        proposal_filtering=False,
        proposal_score_thresh=0.2,
        minimal_proposal_number=100,
        depth_with_uncertainty=False,
        # loss settings
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0
        ),
        loss_bbox=dict(type="SmoothL1Loss", beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
        loss_attr=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
        loss_centerness=dict(
            type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0
        ),
        loss_bbox2d=dict(type="IOULoss", loss_type="giou"),
        bbox_code_size=9,  # For nuscenes
        pred_attrs=False,
        num_attrs=9,  # For nuscenes
        cls_branch=(128, 64),
        cls_agnostic=False,
        reg_keys=["offset", "depth", "size", "rot", "velo"],
        reg_branch=(
            (128, 64, 2),  # offset
            (128, 64, 1),  # depth
            (64, 3),  # size
            (64, 1),  # rot
            (2,),  # velo
        ),
        reg_weights=(),
        dir_branch=(64,),
        attr_branch=(64,),
        # centerness
        centerness_branch=(64,),
        centerness_on_reg=True,
        centerness_alpha=2.5,
        conv_cfg=None,
        norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
        train_cfg=None,
        test_cfg=None,
        init_cfg=None,
        sync_avg_factors=False,
        teacher_force_cfg=None,
        debug=False,
    ):
        super(FCOSMono3D_ProposalHead, self).__init__(init_cfg=init_cfg)

        self.cls_agnostic = cls_agnostic

        # cls_agnostic should be ablated
        if self.cls_agnostic:
            self.num_classes = 1
            self.cls_out_channels = 1
        else:
            self.num_classes = num_classes
            self.cls_out_channels = num_classes

        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == "auto" or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias

        # whether use orientation estimation as an auxiliary branch
        self.use_direction_classifier = use_direction_classifier
        self.diff_rad_by_sin = diff_rad_by_sin
        self.dir_offset = dir_offset

        # loss functions
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_dir = build_loss(loss_dir)
        self.loss_bbox2d = build_loss(loss_bbox2d)
        self.loss_centerness = build_loss(loss_centerness)

        self.sync_avg_factors = sync_avg_factors & dist.is_initialized()

        self.cls_branch = cls_branch
        self.reg_branch = reg_branch
        self.reg_keys = reg_keys
        assert len(reg_keys) == len(reg_weights)
        self.reg_weights = {
            reg_key: reg_weight for reg_key, reg_weight in zip(reg_keys, reg_weights)
        }

        assert len(reg_branch) == len(self.reg_keys), (
            "The number of " "element in reg_branch and reg_keys should be the same."
        )

        self.positive_reg_keys = ["depth", "size", "bbox2d"]
        # only a subset of reg_keys is needed for inference
        self.infernce_reg_keys = ["offset", "depth"]

        self.depth_with_uncertainty = depth_with_uncertainty
        self.dir_branch = dir_branch
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.background_label = (
            self.num_classes if background_label is None else background_label
        )

        # background_label should be either 0 or self.num_classes
        assert self.background_label == 0 or self.background_label == self.num_classes

        # auxiliary branches to be ablated
        self.bbox_code_size = bbox_code_size
        self.pred_attrs = pred_attrs
        self.attr_background_label = -1
        self.num_attrs = num_attrs
        if self.pred_attrs:
            self.attr_background_label = num_attrs
            self.loss_attr = build_loss(loss_attr)
            self.attr_branch = attr_branch

        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.centerness_alpha = centerness_alpha
        self.centerness_branch = centerness_branch

        self.using_2d_centers = using_2d_centers

        # number of proposals for surround-view
        self.num_proposal = num_proposal
        self.num_cam = 6
        self.num_proposal_per_image = self.num_proposal // self.num_cam
        self.generate_proposal_per_image = generate_proposal_per_image  # ablation
        self.objectness_with_centerness = objectness_with_centerness  # True
        self.objectness_max_pooling = objectness_max_pooling  # ablation

        self.generate_multi_class_proposals = generate_multi_class_proposals

        # modify predicted objectness with ground-truth so that better proposals are generated
        self.random_objectness_with_teacher = random_objectness_with_teacher
        self.teacher_force_cfg = teacher_force_cfg

        # proposal filtering
        self.proposal_filtering = proposal_filtering
        self.proposal_score_thresh = proposal_score_thresh
        self.minimal_proposal_number = minimal_proposal_number

        # proposal dropping when generating proposals for two-stage detr
        self.random_proposal_drop = random_proposal_drop
        self.random_proposal_drop_upper_bound = random_proposal_drop_upper_bound
        self.random_proposal_drop_lower_bound = random_proposal_drop_lower_bound

        self._init_layers()
        if init_cfg is None:
            self.init_cfg = dict(
                type="Normal",
                layer="Conv2d",
                std=0.01,
                override=dict(type="Normal", name="conv_cls", std=0.01, bias_prob=0.01),
            )
        else:
            self.init_cfg = init_cfg

        self.target_keys = [
            "bboxes",
            "labels",
            "gt_bboxes_3d",
            "gt_corners_2d",
            "gt_labels_3d",
            "centers2d",
            "depths",
            "attr_labels",
            "match_idxs",
        ]
        self.debug = debug

    def _init_layers(self):
        """Initialize layers of the head."""

        self._init_cls_convs()
        self._init_reg_convs()
        self._init_predictor()

        self.conv_centerness_prev = self._init_branch(
            conv_channels=self.centerness_branch,
            conv_strides=(1,) * len(self.centerness_branch),
        )
        self.conv_centerness = nn.Conv2d(self.centerness_branch[-1], 1, 1)

        """
        separate scale for every specific task in each feature-level
        default scales for offset, depth, size, (bbox_2d), (keypoints)
        """

        scale_related_attributes = [
            "offset",
            "depth",
            "bbox2d",
            "corners",
        ]

        self.scales = nn.ModuleList()
        for _ in self.strides:
            level_scales = nn.ModuleDict()
            for attr in scale_related_attributes:
                if attr in self.reg_keys:
                    level_scales[attr] = Scale(1.0)

            self.scales.append(level_scales)

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type="DCNv2")
            else:
                conv_cfg = self.conv_cfg
            
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias,
                )
            )
        self.cls_convs = nn.Sequential(*self.cls_convs)

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type="DCNv2")
            else:
                conv_cfg = self.conv_cfg
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias,
                )
            )

        self.reg_convs = nn.Sequential(*self.reg_convs)

    def _init_branch(self, conv_channels=(64), conv_strides=(1)):
        """Initialize conv layers as a prediction branch."""
        conv_before_pred = nn.ModuleList()
        if isinstance(conv_channels, int):
            conv_channels = [self.feat_channels] + [conv_channels]
            conv_strides = [conv_strides]
        else:
            conv_channels = [self.feat_channels] + list(conv_channels)
            conv_strides = list(conv_strides)

        for i in range(len(conv_strides)):
            conv_before_pred.append(
                ConvModule(
                    conv_channels[i],
                    conv_channels[i + 1],
                    3,
                    stride=conv_strides[i],
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias,
                )
            )

        conv_before_pred = nn.Sequential(*conv_before_pred)

        return conv_before_pred

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.conv_cls_prev = self._init_branch(
            conv_channels=self.cls_branch, conv_strides=(1,) * len(self.cls_branch)
        )
        self.conv_cls = nn.Conv2d(self.cls_branch[-1], self.cls_out_channels, 1)

        self.conv_reg_prevs = nn.ModuleDict()
        self.conv_regs = nn.ModuleDict()

        for reg_key, reg_branch in zip(self.reg_keys, self.reg_branch):
            reg_dim = reg_branch[-1]
            reg_branch_channels = reg_branch[:-1]

            if len(reg_branch_channels) > 0:
                self.conv_reg_prevs[reg_key] = self._init_branch(
                    conv_channels=reg_branch_channels,
                    conv_strides=(1,) * len(reg_branch_channels),
                )
                self.conv_regs[reg_key] = nn.Conv2d(reg_branch_channels[-1], reg_dim, 1)
            else:
                self.conv_reg_prevs[reg_key] = nn.Identity()
                self.conv_regs[reg_key] = nn.Conv2d(self.feat_channels, reg_dim, 1)

        if self.use_direction_classifier:
            self.conv_dir_cls_prev = self._init_branch(
                conv_channels=self.dir_branch, conv_strides=(1,) * len(self.dir_branch)
            )
            self.conv_dir_cls = nn.Conv2d(self.dir_branch[-1], 2, 1)

        if self.pred_attrs:
            self.conv_attr_prev = self._init_branch(
                conv_channels=self.attr_branch,
                conv_strides=(1,) * len(self.attr_branch),
            )
            self.conv_attr = nn.Conv2d(self.attr_branch[-1], self.num_attrs, 1)

    def init_weights(self):
        super().init_weights()
        bias_cls = bias_init_with_prob(0.01)
        if self.use_direction_classifier:
            normal_init(self.conv_dir_cls, std=0.01, bias=bias_cls)
        if self.pred_attrs:
            normal_init(self.conv_attr, std=0.01, bias=bias_cls)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level,
                    each is a 4D-tensor, the channel number is
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * bbox_code_size.
                dir_cls_preds (list[Tensor]): Box scores for direction class
                    predictions on each scale level, each is a 4D-tensor,
                    the channel number is num_points * 2. (bin = 2).
                attr_preds (list[Tensor]): Attribute scores for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * num_attrs.
                centernesses (list[Tensor]): Centerness for each scale level,
                    each is a 4D-tensor, the channel number is num_points * 1.
        """

        # todo: multi_apply for dict
        fcos_outputs = multi_apply_dic(
            self.forward_single, feats, self.scales, self.strides
        )

        return fcos_outputs

    def get_loss(self, fcos_outputs, mono_targets, img_metas,
            global_gt_bboxes_3d=None, raw_img=None):

        losses, train_utils = self.loss(
            predictions=fcos_outputs,
            targets=mono_targets,
            img_metas=img_metas,
            raw_img=raw_img,
        )

        return losses, train_utils

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale levle.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox and direction class \
                predictions, centerness predictions of input feature maps.
        """

        res = {}
        # merge num_batch & num_cam
        x = x.flatten(start_dim=0, end_dim=1)

        # cls forward
        cls_feat = self.cls_convs(x)
        spec_cls_feat = self.conv_cls_prev(cls_feat)
        cls_score = self.conv_cls(spec_cls_feat)

        res["cls_feat"] = cls_feat.clone()
        res["cls_score"] = cls_score

        # reg forward
        reg_feat = self.reg_convs(x)
        res["reg_feat"] = reg_feat.clone()
        for reg_key in self.conv_reg_prevs.keys():
            if (not self.training) and (reg_key not in self.infernce_reg_keys):
                continue

            this_reg_feat = self.conv_reg_prevs[reg_key](reg_feat)
            res[reg_key] = self.conv_regs[reg_key](this_reg_feat)

            if reg_key in scale:
                res[reg_key] = scale[reg_key](res[reg_key]).float()

            if reg_key in self.positive_reg_keys:
                res[reg_key] = res[reg_key].exp()

        if self.use_direction_classifier and self.training:
            direction_reg_feat = self.conv_dir_cls_prev(reg_feat)
            dir_cls_pred = self.conv_dir_cls(direction_reg_feat)
            res["dir_cls"] = dir_cls_pred

        if self.pred_attrs and self.training:
            attr_cls_feat = self.conv_attr_prev(cls_feat)
            attr_pred = self.conv_attr(attr_cls_feat)
            res["attr"] = attr_pred

        if self.centerness_on_reg:
            centerness_feat = self.conv_centerness_prev(reg_feat)
            res["centerness"] = self.conv_centerness(centerness_feat)
        else:
            centerness_feat = self.conv_centerness_prev(cls_feat)
            res["centerness"] = self.conv_centerness(centerness_feat)

        assert self.norm_on_bbox is True, (
            "Setting norm_on_bbox to False "
            "has not been thoroughly tested for FCOS3D."
        )

        return res

    def visualize_cls_centerness(self, img_metas, featmap_sizes, fcos_targets, fcos_outputs, raw_img=None):
        import imageio
        import os
        import matplotlib.pyplot as plt

        visualize_path = "debugs/fcos_cls"
        os.makedirs(visualize_path, exist_ok=True)

        def overlay_img_with_proposals(img, mask, color=(0, 255, 0)):
            if tuple(mask.shape) != img.shape[:2]:
                mask = F.interpolate(mask[None, None].float(), size=img.shape[:2], mode='bilinear').squeeze()

            mask = mask.detach().cpu().numpy()
            colorful_mask = mask[..., np.newaxis] * np.array(color)
            res = img * (1 - mask[..., np.newaxis]) + colorful_mask

            return res.astype(np.uint8)
        
        def visualize_multi_level(raw_img, targets, preds, save_file='*.png'):
            num_level = len(targets)
            
            plt.figure(figsize=(64, 18))
            for feat_level, level_targets in enumerate(targets):
                plt.subplot(2, num_level, feat_level + 1)
                plt.imshow(overlay_img_with_proposals(raw_img, level_targets)); plt.axis("off")
                
                plt.subplot(2, num_level, feat_level + num_level + 1);
                plt.imshow(overlay_img_with_proposals(raw_img, preds[feat_level])); plt.axis("off")
            
            plt.tight_layout()
            plt.savefig(save_file)
            plt.close()
        
        ''' process training targets '''

        labels_3d = fcos_targets["labels_3d"]
        centerness_targets = fcos_targets["centerness"]
        # reshape to map
        labels_3d = [x.view(6, feat_size[0], feat_size[1]) for x, feat_size in zip(labels_3d, featmap_sizes)]
        centerness_targets = [x.view(6, feat_size[0], feat_size[1]) for x, feat_size in zip(centerness_targets, featmap_sizes)]
        # preprocess
        foreground_labels_3d = [x < self.num_classes for x in labels_3d]

        pred_cls_score = [torch.max(x['cls_score'], dim=1)[0].sigmoid() for x in fcos_outputs]
        pred_centerness_score = [x['centerness'].squeeze(dim=1).sigmoid() for x in fcos_outputs]
        pred_objectness_score = [x * y for x, y in zip(pred_cls_score, pred_centerness_score)]

        for batch_id in range(len(img_metas)):
            img_filenames = img_metas[batch_id]["filename"]
            for cam_index, img_filename in enumerate(img_filenames):
                img = raw_img[batch_id][cam_index].cpu().numpy().astype(np.uint8)[..., [2, 1, 0]]

                # visualize cls
                visualize_multi_level(
                    raw_img=img,
                    targets=[x[cam_index] for x in foreground_labels_3d],
                    preds=[x[cam_index] for x in pred_cls_score],
                    save_file='{}/cls_cam{}.png'.format(visualize_path, cam_index),
                )

                # visualize centerness
                visualize_multi_level(
                    raw_img=img,
                    targets=[x[cam_index] for x in centerness_targets],
                    preds=[x[cam_index] for x in pred_objectness_score],
                    save_file='{}/centerness_cam{}.png'.format(visualize_path, cam_index),
                )
        
        print("[DEBUG]: visualize the predicted and ground-truth classification & centerness")
        pdb.set_trace()
        
    def loss(
        self,
        predictions,
        targets,
        img_metas,
        gt_bboxes_ignore=None,
        raw_img=None,
    ):
        """Compute loss of the head.

        Args:
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        featmap_sizes = [pred["cls_score"].size()[-2:] for pred in predictions]
        # list with size of num_level, each element contains all the coordinates (N x 2) in the feature map
        all_level_points = self.get_points(
            featmap_sizes,
            predictions[0]["cls_score"].dtype,
            predictions[0]["cls_score"].device,
        )

        training_targets = self.get_targets(points=all_level_points, targets=targets)
        train_utils = training_targets.copy()
        training_targets.pop("gt_match_idxs")
        train_utils["featmap_sizes"] = featmap_sizes

        # visualize predictions and classifications
        if self.debug:
            self.visualize_cls_centerness(img_metas, featmap_sizes, training_targets, predictions, raw_img=raw_img)

        # flatten predictions
        num_imgs = predictions[0]["cls_score"].shape[0]

        flatten_preds = {}
        for pred_key in predictions[0].keys():
            flatten_preds[pred_key] = torch.cat(
                [
                    pred[pred_key].permute(0, 2, 3, 1).flatten(0, 2)
                    for pred in predictions
                ]
            )

        for key, val in training_targets.items():
            training_targets[key] = torch.cat(val)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        flatten_labels_3d = training_targets.pop("labels_3d")
        pos_inds = (
            ((flatten_labels_3d >= 0) & (flatten_labels_3d < bg_class_ind))
            .nonzero()
            .reshape(-1)
        )
        # radius of 1.5 ==> 3x3 neighbouring area
        num_pos = torch.tensor(len(pos_inds)).type_as(flatten_preds["cls_score"])
        loss_cls = self.loss_cls(
            flatten_preds.pop("cls_score"),
            flatten_labels_3d,
            avg_factor=torch.clamp_min(num_pos, 1.0),
        )

        # positive predictions
        for key, val in flatten_preds.items():
            flatten_preds[key] = val[pos_inds]

        # positive targets
        for key, val in training_targets.items():
            training_targets[key] = val[pos_inds]

        """
        pred_keys = dict_keys(['cls_feat', 'offset', 'depth', 'size', 'rot', 'bbox2d', 'corners', 'velo', 'reg_feat', 'dir_cls', 'attr', 'centerness'])

        target_keys = dict_keys(['attr_labels', 'offset', 'depth', 'size', 'rot', 'bbox2d', 'corners', 'velo', 'centerness'])
        """

        loss_dict = {"cls": loss_cls}
        equal_weights = pos_inds.new_ones(pos_inds.shape).float()
        pos_centerness_targets = training_targets["centerness"]

        if num_pos > 0:
            # True
            if self.use_direction_classifier:
                pos_rot_targets = training_targets["rot"].squeeze(dim=1)
                pos_dir_cls_targets = self.get_direction_target(
                    pos_rot_targets, dir_offset=self.dir_offset, one_hot=False
                )

            # True
            if self.diff_rad_by_sin and "rot" in self.reg_keys:
                pos_rot_preds, pos_rot_targets = self.add_sin_difference(
                    flatten_preds["rot"], training_targets["rot"]
                )

                flatten_preds["rot"] = pos_rot_preds
                training_targets["rot"] = pos_rot_targets

            avg_factor = equal_weights.sum()
            for reg_key in self.reg_keys:
                # for bbox2d, IoU loss is adopted
                if reg_key == "bbox2d":
                    loss_dict[reg_key] = self.loss_bbox2d(
                        flatten_preds.pop(reg_key),
                        training_targets.pop(reg_key),
                        avg_factor=avg_factor,
                    )
                    loss_dict[reg_key] = loss_dict[reg_key] * self.reg_weights[reg_key]
                    continue

                if reg_key == "depth" and self.depth_with_uncertainty:
                    pred_depth_with_uncertainty = flatten_preds.pop(reg_key)
                    target_depth = training_targets.pop(reg_key)
                    pred_depth, pred_uncertainty = torch.split(
                        pred_depth_with_uncertainty, [1, 1], dim=1
                    )

                    uncertainty_weight = 1.41421 / pred_uncertainty.exp()
                    loss_dict[reg_key] = self.loss_bbox(
                        pred_depth,
                        target_depth,
                        weight=uncertainty_weight,
                        avg_factor=avg_factor,
                    )
                    loss_dict[reg_key] += pred_uncertainty.sum() / avg_factor
                    loss_dict[reg_key] *= self.reg_weights[reg_key]
                    continue

                if reg_key == "corners":
                    key_preds = flatten_preds.pop(reg_key)
                    key_targets = training_targets.pop(reg_key)

                    corners_valid_mask = (key_targets[..., -8:] > 0).float()
                    key_avg_factor = corners_valid_mask.sum()
                    # rep & multiply
                    corners_valid_mask = corners_valid_mask.repeat(1, 2)
                    key_preds *= corners_valid_mask
                    key_targets = key_targets[..., :16] * corners_valid_mask
                else:
                    key_avg_factor = avg_factor
                    key_preds = flatten_preds.pop(reg_key)
                    key_targets = training_targets.pop(reg_key)

                loss_dict[reg_key] = self.loss_bbox(
                    key_preds,
                    key_targets,
                    weight=self.reg_weights[reg_key],
                    avg_factor=key_avg_factor,
                )

            loss_dict["centerness"] = self.loss_centerness(
                flatten_preds["centerness"].squeeze(dim=1),
                training_targets["centerness"],
            )

            # direction classification loss
            if self.use_direction_classifier:
                loss_dict["dir_cls"] = self.loss_dir(
                    flatten_preds["dir_cls"],
                    pos_dir_cls_targets,
                    equal_weights,
                    avg_factor=avg_factor,
                )

            # attribute classification loss
            if self.pred_attrs:
                loss_dict["attr"] = self.loss_attr(
                    flatten_preds["attr"],
                    training_targets["attr_labels"],
                    pos_centerness_targets,
                    avg_factor=pos_centerness_targets.sum(),
                )
        else:
            # need absolute due to possible negative delta x/y
            for reg_key in self.reg_keys:
                loss_dict[reg_key] = flatten_preds[reg_key].abs().sum()

            loss_dict["centerness"] = flatten_preds["centerness"].sum()

            if self.use_direction_classifier:
                loss_dict["dir_cls"] = flatten_preds["dir_cls"].sum()

            if self.pred_attrs:
                loss_dict["attr"] = flatten_preds["attr"].sum()

        return loss_dict, train_utils
    
    def teacher_forcing(self, objness_pred, fcos_utils):
        teacher_prob = self.random_objectness_with_teacher
        if self.teacher_force_cfg is not None:
            # decay based on epochs
            if self.teacher_force_cfg['epoch_based']:
                decay_mu = self.teacher_force_cfg['decay_mu']
                decay_rate = decay_mu / (decay_mu + np.exp(self.epoch / self.max_epochs))
            else:
                raise NotImplementedError
            
            teacher_prob *= decay_rate

        if np.random.rand() > teacher_prob:
            return objness_pred

        labels_3d = fcos_utils["labels_3d"]
        ctr_3d = fcos_utils['centerness']
        for fpn_level in range(len(objness_pred)):
            # foreground mask
            has_gt = labels_3d[fpn_level] != self.num_classes

            # foreground targets
            objness_target = has_gt.float() * ctr_3d[fpn_level]
            objness_target = objness_target.view(objness_pred[fpn_level].shape)

            objness_pred[fpn_level] = torch.where(
                has_gt.view(objness_pred[fpn_level].shape),
                objness_target,
                objness_pred[fpn_level],
            )
        
        return objness_pred

    def get_proposals(
        self,
        fcos_outputs,
        fcos_utils=None,
        img_metas=None,
        gt_bboxes_3d=None,
        raw_img=None,
    ):
        """Generate proposals from one-stage classification results

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W)
            dir_cls_preds (list[Tensor]): Box scores for direction class
                predictions on each scale level, each is a 4D-tensor,
                the channel number is num_points * 2. (bin = 2)
            attr_preds (list[Tensor]): Attribute scores for each scale level
                Has shape (N, num_points * num_attrs, H, W)
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W)

        Returns:
            proposals (Tensor): object proposals for full-surround sample from all views & scales with shape (batch, num_proposal)
        """

        # fcos_outputs, list of dict with keys = dict_keys(['cls_feat', 'cls_score', 'offset', 'depth', 'size', 'rot', 'bbox2d', 'corners', 'velo', 'reg_feat', 'dir_cls', 'attr', 'centerness'])

        # fcos_utils: dict with keys = dict_keys(['labels_3d', 'attr_labels', 'offset', 'depth', 'size', 'rot', 'bbox2d', 'corners', 'velo', 'centerness', 'featmap_sizes'])

        cls_score = [output["cls_score"] for output in fcos_outputs]
        centerness = [output["centerness"] for output in fcos_outputs]

        if self.generate_multi_class_proposals:
            # list of [B, num_class, H', W']
            objness_pred = [logits.sigmoid() for logits in cls_score]
        else:
            objness_pred = [
                logits.sigmoid().max(dim=1, keepdim=True)[0] for logits in cls_score
            ]
        
        # ''' recall test: using ground-truth scores '''
        # gt_cls_score = fcos_utils['labels_3d']
        # gt_objness = [(x != self.num_classes).float() for x in gt_cls_score]
        # objness_pred = [x.view(y.shape) for x, y in zip(gt_objness, objness_pred)]

        # ''' recall test: using ground-truth centerness '''
        # objness_pred = [
        #         obj * ctr.view(obj.shape) for obj, ctr in zip(objness_pred, fcos_utils['centerness'])
        #     ]

        if self.objectness_with_centerness:
            objness_pred = [
                obj * ctr.sigmoid() for obj, ctr in zip(objness_pred, centerness)
            ]

        if self.objectness_max_pooling:
            for index, level_objness in enumerate(objness_pred):
                level_objness_nms = nn.functional.max_pool2d(
                    level_objness, (3, 3), stride=1, padding=1
                )
                objness_pred[index] = (
                    level_objness * (level_objness == level_objness_nms).float()
                )
        
        # summation after the class-wise max-pooling
        if self.generate_multi_class_proposals:
            objness_pred = [scores.sum(dim=1, keepdim=True) for scores in objness_pred]
        
        if self.training and self.random_objectness_with_teacher > 0:
            objness_pred = self.teacher_forcing(objness_pred, fcos_utils)

        scores = []
        transformer_features = []
        position_3d = []
        fpn_levels = []

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_score]
        # list with size of num_level, each element contains all the coordinates (N x 2) in the feature map
        all_level_points = self.get_points(featmap_sizes, cls_score[0].dtype, cls_score[0].device)

        batch_size = len(img_metas)
        # batch_size = cls_score[0].size(0)
        cam_intrinsics = []
        lidar2cams = []
        # print(img_metas)

        for img_metas_k in img_metas:
            if isinstance(img_metas_k["intrinsics"], DC):
                cam_intrinsics.extend(img_metas_k["intrinsics"].data)
                lidar2cams.extend(img_metas_k["extrinsics"].data)
            else:
                cam_intrinsics.extend(img_metas_k["intrinsics"])
                lidar2cams.extend(img_metas_k["extrinsics"])
            # cam_intrinsics.extend(img_metas_k["cam2img"])
            # lidar2cams.extend(img_metas_k["lidar2cam"])

        cam_intrinsics = np.stack(cam_intrinsics, axis=0)
        lidar2cams = np.stack(lidar2cams, axis=0)
        cam_intrinsics = cls_score[0].new_tensor(cam_intrinsics).float()
        lidar2cams = cls_score[0].new_tensor(lidar2cams).float()

        if self.training:
            gt_match_idxs = fcos_utils["gt_match_idxs"]
            combined_gt_match_idxs = []

        # can we visualize the positions of selected proposals
        for fpn_level in range(len(objness_pred)):
            objness = objness_pred[fpn_level]
            pred_offset = fcos_outputs[fpn_level]["offset"]
            pred_depth = fcos_outputs[fpn_level]["depth"]
            cls_tower = fcos_outputs[fpn_level]["cls_feat"]
            bbox_tower = fcos_outputs[fpn_level]["reg_feat"]
            level_points = all_level_points[fpn_level]

            b, _, h, w = objness.shape
            joint_feature = torch.cat([cls_tower, bbox_tower], dim=1)
            joint_feature = joint_feature.flatten(2).permute(0, 2, 1)  # [b, h * w, c]
            objness = objness.view(b, h * w)  # # [b, h * w]

            pred_offset = pred_offset.flatten(2).permute(0, 2, 1)
            pred_offset *= self.strides[fpn_level]
            pred_center2d = level_points.unsqueeze(dim=0) - pred_offset

            if self.depth_with_uncertainty:
                pred_depth = pred_depth[:, 0:1]

            # convert 2.5D centers ==> 3D centers in CameraCoordinates
            pred_depths = pred_depth.flatten(2).permute(0, 2, 1)
            homo_pred_center2d = torch.cat(
                (
                    pred_center2d[..., :2] * pred_depths,
                    pred_depths,
                    pred_depths.new_ones(pred_depths.shape),
                ),
                dim=-1,
            )

            inv_K = torch.inverse(cam_intrinsics).transpose(1, 2)
            points3d_cam_homo = torch.bmm(homo_pred_center2d, inv_K)

            # convert 3D centers in CameraCoordinates ==> 3D centers in LiDARCoordinates
            cam2lidars = torch.inverse(lidar2cams).transpose(1, 2)
            points3d_lidar = torch.bmm(points3d_cam_homo, cam2lidars)[..., :3]

            if self.training:
                level_gt_match_idxs = gt_match_idxs[fpn_level]
                level_gt_match_idxs = level_gt_match_idxs.view(b, h * w)
                combined_gt_match_idxs.append(level_gt_match_idxs)
            scores.append(objness)
            transformer_features.append(joint_feature)
            position_3d.append(points3d_lidar)
            fpn_levels.append(
                torch.full(
                    objness.shape, fpn_level, dtype=torch.int64, device=objness.device
                )
            )

        scores = torch.cat(scores, dim=1)
        transformer_features = torch.cat(transformer_features, dim=1)
        fpn_levels = torch.cat(fpn_levels, dim=1)
        position_3d = torch.cat(position_3d, dim=1)
        view_indicators = (
            torch.arange(self.num_cam * batch_size).view(-1, 1).expand_as(scores).type_as(scores)
        )

        scores = scores.view(batch_size, -1)
        num_total_proposal_per_batch = scores.shape[-1]
        _, poi_idx = torch.topk(
            scores,
            self.num_proposal,
            sorted=True,
            dim=1,
        )
        # print(scores.size(), poi_idx.size())
        num_view = cls_score[0].size(0)
        poi_idx_flatten = (poi_idx + torch.arange(batch_size).type_as(poi_idx).view(-1, 1) * num_total_proposal_per_batch)
        # poi_idx_flatten = poi_idx_flatten.view(-1)
        # print(poi_idx_flatten.size(), scores.max())

        if self.proposal_filtering: 
            poi_scores = torch.gather(scores, 1, poi_idx)
            # print(poi_scores.size())
            valid_mask = poi_scores >= self.proposal_score_thresh
            # print(valid_mask.sum(1))
            minimal_proposal_number = max(max(valid_mask.sum(1)), self.minimal_proposal_number)
            # print(valid_mask.sum(1), minimal_proposal_number)
            poi_idx_flatten = poi_idx_flatten[:, :minimal_proposal_number].contiguous()
        poi_idx_flatten = poi_idx_flatten.view(-1)

        # # filter poi_idx with score threshold
        # if self.proposal_filtering:
        #     poi_scores = scores.view(-1)[poi_idx_flatten]
        #     print(poi_scores.size())
        #     valid_mask = poi_scores >= self.proposal_score_thresh
            
        #     if valid_mask.sum().item() > self.minimal_proposal_number:
        #         poi_idx_flatten = poi_idx_flatten[valid_mask]
        #     else:
        #         poi_idx_flatten = poi_idx_flatten[:self.minimal_proposal_number]

        # random sampling of proposals
        if self.random_proposal_drop and self.training:
            diff = (
                self.random_proposal_drop_upper_bound
                - self.random_proposal_drop_lower_bound
            )
            sample_factor = (
                self.random_proposal_drop_upper_bound - np.random.rand(1)[0] * diff
            )

            num_generated_proposal = poi_idx_flatten.shape[0]
            num_input_proposal = int(num_generated_proposal * sample_factor)
            # sampling proposals
            subsample_idxs = np.random.choice(
                num_generated_proposal,
                num_input_proposal,
                replace=False,
            )
            subsample_idxs = torch.from_numpy(subsample_idxs).to(poi_idx.device)
            poi_idx_flatten = poi_idx_flatten[subsample_idxs]
        

        """
        Visualize the positions of selected proposals
        """
        if self.debug:
            self.visualize_proposal(
                scores,
                poi_idx_flatten,
                all_level_points,
                featmap_sizes,
                img_metas,
                batch_size,
                raw_img=raw_img,
            )

            print('[DEBUG]: visualize the classification results & the positions of proposals')
        
        '''
        prepare proposal information for the second-stage, including:
        1. proposal_features
        2. positional information (3D position, from which level, from which camera)
        3. proposal match indices to filter corresponding GT bboxes_3D
        4. proposal scores for two-stage classification combination
        
         '''

        res_proposals = {}
        proposal_transformer_features = transformer_features.view(-1, transformer_features.shape[-1])[poi_idx_flatten]
        proposal_position_3d = position_3d.view(-1, position_3d.shape[-1])[poi_idx_flatten]
        proposal_view_indicators = view_indicators.view(-1)[poi_idx_flatten]
        proposal_fpn_levels = fpn_levels.view(-1)[poi_idx_flatten]
        proposal_position_3d = torch.cat((proposal_position_3d, proposal_view_indicators.unsqueeze(dim=1), proposal_fpn_levels.unsqueeze(dim=1)), dim=1)

        proposal_scores = scores.view(-1)[poi_idx_flatten]

        num_in_proposal = poi_idx_flatten.shape[0] // batch_size      
        proposal_transformer_features = proposal_transformer_features.view(batch_size, num_in_proposal, -1)
        proposal_position_3d = proposal_position_3d.view(batch_size, num_in_proposal, -1)
        proposal_scores = proposal_scores.view(batch_size, num_in_proposal)

        res_proposals = {
            'proposal_features': proposal_transformer_features,
            'proposal_positions': proposal_position_3d,
            'proposal_scores': proposal_scores,
        }

        if self.training:
            combined_gt_match_idxs = torch.cat(combined_gt_match_idxs, dim=1)
            poi_match_idxs = combined_gt_match_idxs.view(-1)[poi_idx_flatten]
            poi_match_idxs = poi_match_idxs.long()
            
            uni_poi_match_idxs = torch.unique(poi_match_idxs)
            uni_poi_match_idxs = uni_poi_match_idxs[uni_poi_match_idxs >= 0]
            res_proposals['proposal_match_idxs'] = uni_poi_match_idxs.long()
            # -1 for background, [0, ..., N] refers to objects

            updated_poi_match_idxs = ((poi_match_idxs.unsqueeze(1) - uni_poi_match_idxs.unsqueeze(0)) == 0).nonzero()
            updated_poi_match_idxs = updated_poi_match_idxs.long()
            poi_match_idxs[updated_poi_match_idxs[:, 0]] = updated_poi_match_idxs[:, 1]
            res_proposals['proposal_corr_idxs'] = poi_match_idxs.view(batch_size, -1).long()
        
        return res_proposals

    def get_targets(
        self,
        points,
        targets,
    ):
        """Compute regression, classification and centerss targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
            gt_bboxes_3d_list (list[Tensor]): 3D Ground truth bboxes of each
                image, each has shape (num_gt, bbox_code_size).
            gt_labels_3d_list (list[Tensor]): 3D Ground truth labels of each
                box, each has shape (num_gt,).
            centers2d_list (list[Tensor]): Projected 3D centers onto 2D image,
                each has shape (num_gt, 2).
            depths_list (list[Tensor]): Depth of projected 3D centers onto 2D
                image, each has shape (num_gt, 1).
            attr_labels_list (list[Tensor]): Attribute labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """

        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)

        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(points[i])
            for i in range(num_levels)
        ]

        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        training_targets = multi_apply_dic(
            self._get_target_single,
            targets,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points,
        )

        target_keys = training_targets[0].keys()
        for view_index, training_target in enumerate(training_targets):
            for target_key in target_keys:
                training_targets[view_index][target_key] = training_target[
                    target_key
                ].split(num_points, 0)

        concat_training_targets = defaultdict(list)
        for level_index in range(num_levels):
            for target_key in target_keys:
                level_training_targets = torch.cat(
                    [
                        view_targets[target_key][level_index]
                        for view_targets in training_targets
                    ]
                )

                # normalize corners
                if target_key in ["offset", "bbox2d", "corners"]:
                    level_training_targets /= self.strides[level_index]

                concat_training_targets[target_key].append(level_training_targets)

        return concat_training_targets

    def _get_target_single(
        self,
        targets,
        points,
        regress_ranges,
        num_points_per_lvl,
    ):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        # num_gts = targets["labels"][0].size(0)
        # print(targets["labels"][0].shape)
        # print(targets["labels"])
        num_gts = targets["labels"].shape[0]

        if num_gts == 0:
            return {
                "labels_3d": points.new_full(
                    (num_points,), self.background_label
                ).long(),
                "attr_labels": points.new_full(
                    (num_points,), self.attr_background_label
                ).long(),
                "offset": points.new_zeros((num_points, 2)).float(),
                "depth": points.new_zeros((num_points, 1)).float(),
                "size": points.new_zeros((num_points, 3)).float(),
                "rot": points.new_zeros((num_points, 1)).float(),
                "bbox2d": points.new_zeros((num_points, 4)).float(),
                "corners": points.new_zeros((num_points, 24)).float(),
                "velo": points.new_zeros((num_points, 2)).float(),
                "centerness": points.new_zeros((num_points,)).float(),
                "gt_match_idxs": points.new_zeros((num_points,)).float(),
            }

        # gt_bboxes = targets["bboxes"][0]
        # gt_bboxes_3d = targets["gt_bboxes_3d"][0]
        # centers2d = targets["centers2d"][0]
        # depths = targets["depths"][0]
        # gt_corners_2d = targets["gt_corners_2d"][0]
        # gt_labels = targets["labels"][0]
        # gt_labels_3d = targets["gt_labels_3d"][0]
        # attr_labels = targets["attr_labels"][0]
        # match_idxs = targets["match_idxs"][0]

        gt_bboxes = targets["bboxes"].to(points.device)
        gt_bboxes_3d = targets["gt_bboxes_3d"].to(points.device)
        centers2d = targets["centers2d"].to(points.device)
        depths = targets["depths"].to(points.device)
        gt_corners_2d = targets["gt_corners_2d"].to(points.device)
        gt_labels = targets["labels"].to(points.device)
        gt_labels_3d = targets["gt_labels_3d"].to(points.device)
        attr_labels = targets["attr_labels"].to(points.device)
        match_idxs = torch.from_numpy(targets["match_idxs"]).to(points.device)

        if self.cls_agnostic:
            gt_labels[:] = 0
            gt_labels_3d[:] = 0

        if not isinstance(gt_bboxes_3d, torch.Tensor):
            gt_bboxes_3d = gt_bboxes_3d.tensor.to(gt_bboxes.device)

        # change orientation to local yaw
        gt_bboxes_3d[..., 6] = (
            -torch.atan2(gt_bboxes_3d[..., 0], gt_bboxes_3d[..., 2])
            + gt_bboxes_3d[..., 6]
        )

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1]
        )
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, -1, -1)
        centers2d = centers2d[None].expand(num_points, -1, -1)
        gt_bboxes_3d = gt_bboxes_3d[None].expand(num_points, -1, -1)
        depths = depths[None, :, None].expand(num_points, -1, -1)

        # [num_points, num_objects, 2]
        real_centers2d = (gt_bboxes[..., :2] + gt_bboxes[..., 2:]) / 2

        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, -1)
        ys = ys[:, None].expand(num_points, -1)

        # offsets to the 2.5D centers
        delta_xs = (xs - centers2d[..., 0])[..., None]
        delta_ys = (ys - centers2d[..., 1])[..., None]

        # offsets to the 2.5D centers
        delta_xs_2d = (xs - real_centers2d[..., 0])[..., None]
        delta_ys_2d = (ys - real_centers2d[..., 1])[..., None]
        offsets_centers2d = torch.cat((delta_xs_2d, delta_ys_2d), dim=-1)

        # 2.5D center, depth, dim, orientation, velocity
        bbox_targets_3d = torch.cat(
            (delta_xs, delta_ys, depths, gt_bboxes_3d[..., 3:]), dim=-1
        )

        target_offsets = torch.cat((delta_xs, delta_ys), dim=-1)
        target_depths = depths
        target_sizes = gt_bboxes_3d[..., 3:6]
        target_rots = gt_bboxes_3d[..., 6:7]
        target_velos = gt_bboxes_3d[..., 7:9]

        gt_corners_2d = gt_corners_2d[None].expand(num_points, -1, -1, -1)
        # num_points, num_gts, num_corners
        bbox_corners_delta_xs = xs[..., None] - gt_corners_2d[..., 0]
        bbox_corners_delta_ys = ys[..., None] - gt_corners_2d[..., 1]
        bbox_corners_mask = gt_corners_2d[..., 2]
        target_corners = torch.cat(
            (
                bbox_corners_delta_xs,
                bbox_corners_delta_ys,
                bbox_corners_mask,
            ),
            dim=-1,
        ).float()

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        target_bbox2d = torch.stack((left, top, right, bottom), -1)

        assert self.center_sampling is True, (
            "Setting center_sampling to " "False has not been implemented for FCOS3D."
        )

        # condition1: inside a `center bbox`
        radius = self.center_sample_radius

        if self.using_2d_centers:
            center_xs = real_centers2d[..., 0]
            center_ys = real_centers2d[..., 1]
        else:
            center_xs = centers2d[..., 0]
            center_ys = centers2d[..., 1]
        
        center_gts = torch.zeros_like(gt_bboxes)
        stride = center_xs.new_zeros(center_xs.shape)

        # project the points on current lvl back to the `original` sizes
        lvl_begin = 0
        for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
            lvl_end = lvl_begin + num_points_lvl
            stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
            lvl_begin = lvl_end

        # 中心区域的划分
        center_gts[..., 0] = center_xs - stride
        center_gts[..., 1] = center_ys - stride
        center_gts[..., 2] = center_xs + stride
        center_gts[..., 3] = center_ys + stride

        cb_dist_left = xs - center_gts[..., 0]
        cb_dist_right = center_gts[..., 2] - xs
        cb_dist_top = ys - center_gts[..., 1]
        cb_dist_bottom = center_gts[..., 3] - ys
        center_bbox = torch.stack(
            (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1
        )

        # 像素位于中心区域内
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = target_bbox2d.max(-1)[0]
        inside_regress_range = (max_regress_distance >= regress_ranges[..., 0]) & (
            max_regress_distance <= regress_ranges[..., 1]
        )

        # center-based criterion to deal with ambiguity
        dists = torch.sqrt(torch.sum(bbox_targets_3d[..., :2] ** 2, dim=-1))
        dists[inside_gt_bbox_mask == 0] = INF
        dists[inside_regress_range == 0] = INF
        min_dist, min_dist_inds = dists.min(dim=1)

        labels = gt_labels[min_dist_inds]
        labels_3d = gt_labels_3d[min_dist_inds]
        attr_labels = attr_labels[min_dist_inds]
        labels[min_dist == INF] = self.background_label  # set as BG
        labels_3d[min_dist == INF] = self.background_label  # set as BG
        attr_labels[min_dist == INF] = self.attr_background_label

        # no matched ground-truth object
        gt_match_idxs = match_idxs[min_dist_inds]
        gt_match_idxs[min_dist == INF] = -1

        target_bbox2d = target_bbox2d[range(num_points), min_dist_inds]
        target_offsets = target_offsets[range(num_points), min_dist_inds]
        target_depths = target_depths[range(num_points), min_dist_inds]
        target_sizes = target_sizes[range(num_points), min_dist_inds]
        target_rots = target_rots[range(num_points), min_dist_inds]
        target_velos = target_velos[range(num_points), min_dist_inds]
        target_corners = target_corners[range(num_points), min_dist_inds]

        offsets_centers2d = offsets_centers2d[range(num_points), min_dist_inds]

        if self.using_2d_centers:
            relative_dists = torch.sqrt(torch.sum(offsets_centers2d ** 2, dim=-1)) / (
                1.414 * stride[:, 0]
            )
        else:
            relative_dists = torch.sqrt(torch.sum(target_offsets ** 2, dim=-1)) / (
                1.414 * stride[:, 0]
            )
        # [N, 1] / [N, 1]
        centerness_targets = torch.exp(-self.centerness_alpha * relative_dists)

        training_targets = {
            "labels_3d": labels_3d,
            "attr_labels": attr_labels,
            "offset": target_offsets,
            "depth": target_depths,
            "size": target_sizes,
            "rot": target_rots,
            "bbox2d": target_bbox2d,
            "corners": target_corners,
            "velo": target_velos,
            "centerness": centerness_targets,
            "gt_match_idxs": gt_match_idxs,
        }

        return training_targets

    @staticmethod
    def add_sin_difference(rots1, rots2):
        """Convert the rotation difference to difference in sine function.

        Args:
            boxes1 (torch.Tensor): Original Boxes in shape (NxC), where C>=7
                and the 7th dimension is rotation dimension.
            boxes2 (torch.Tensor): Target boxes in shape (NxC), where C>=7 and
                the 7th dimension is rotation dimension.

        Returns:
            tuple[torch.Tensor]: ``boxes1`` and ``boxes2`` whose 7th \
                dimensions are changed.
        """
        rad_pred_encoding = rots1.sin() * rots2.cos()
        rad_tg_encoding = rots1.cos() * rots2.sin()

        return rad_pred_encoding, rad_tg_encoding

    @staticmethod
    def get_direction_target(rot_gt, dir_offset=0, num_bins=2, one_hot=True):
        """Encode direction to 0 ~ num_bins-1.

        Args:
            reg_targets (torch.Tensor): Bbox regression targets.
            dir_offset (int): Direction offset.
            num_bins (int): Number of bins to divide 2*PI.
            one_hot (bool): Whether to encode as one hot.

        Returns:
            torch.Tensor: Encoded direction targets.
        """
        offset_rot = limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
        if one_hot:
            dir_targets = torch.zeros(
                *list(dir_cls_targets.shape),
                num_bins,
                dtype=rot_gt.dtype,
                device=dir_cls_targets.device
            )
            dir_targets.scatter_(dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets

        return dir_cls_targets

    def _get_points_single(self, featmap_size, stride, dtype, device, flatten=False):
        """Get points of a single scale level."""
        h, w = featmap_size
        x_range = torch.arange(w, dtype=dtype, device=device)
        y_range = torch.arange(h, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range, indexing="ij")

        points = (
            torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride), dim=-1)
            + stride // 2
        )

        return points

    def get_points(self, featmap_sizes, dtype, device, flatten=False):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self._get_points_single(
                    featmap_sizes[i], self.strides[i], dtype, device, flatten
                )
            )

        return mlvl_points
    
    def get_bboxes(self):
        pass

    def visualize_proposal(
        self,
        scores,
        poi_idx_flatten,
        all_level_points,
        featmap_sizes,
        img_metas,
        batch_size,
        raw_img=None,
    ):
        import os
        import imageio
        import matplotlib.pyplot as plt

        save_path = "debugs/proposals"
        os.makedirs(save_path, exist_ok=True)

        proposal_mask = scores.new_zeros(scores.numel())
        proposal_mask[poi_idx_flatten] = 1
        proposal_mask = proposal_mask.view(batch_size * self.num_cam, -1)
        num_points_per_level = [x.shape[0] for x in all_level_points]
        per_level_proposal_masks = torch.split(
            proposal_mask, num_points_per_level, dim=1
        )
        per_level_proposal_masks = [
            mask.view(mask.shape[0], feat_size[0], feat_size[1])
            for mask, feat_size in zip(per_level_proposal_masks, featmap_sizes)
        ]
        input_height, input_width = raw_img.shape[2:4]
        interp_proposal_masks = [F.interpolate(x.unsqueeze(1), size=(input_height, input_width), mode='nearest').squeeze(1) for x in per_level_proposal_masks]
        per_level_proposal_masks = [
            x.detach().cpu().numpy() for x in per_level_proposal_masks
        ]

        def overlay_img_with_proposals(img, mask, color=(0, 255, 0)):
            mask = mask.detach().cpu().numpy()  
            colorful_mask = mask[..., np.newaxis] * np.array(color)
            res = img * (1 - mask[..., np.newaxis]) + colorful_mask

            return res.astype(np.uint8)

        for batch_index in range(batch_size):
            per_img_metas = img_metas[batch_index]
            for cam_index in range(self.num_cam):
                img_filename = per_img_metas["filename"][cam_index]
                img = imageio.imread(img_filename)
                view_index = batch_index * self.num_cam + cam_index

                input_img = raw_img[batch_index, cam_index]
                input_img = input_img.detach().cpu().numpy().astype(np.uint8)

                plt.figure(0, figsize=(24, 9))
                plt.subplot(231)
                plt.imshow(input_img)
                plt.axis("off")
                plt.title('input_image')
                plt.subplot(234)
                plt.imshow(input_img)
                plt.title('input_image')
                plt.axis("off")
                plt.subplot(232)
                plt.imshow(overlay_img_with_proposals(input_img, interp_proposal_masks[0][view_index]))
                # plt.imshow(per_level_proposal_masks[0][view_index], vmin=0.0, vmax=1.0)
                plt.axis("off")
                plt.title('proposal_level_0')
                plt.subplot(233)
                plt.imshow(overlay_img_with_proposals(input_img, interp_proposal_masks[1][view_index]))
                # plt.imshow(per_level_proposal_masks[1][view_index], vmin=0.0, vmax=1.0)
                plt.axis("off")
                plt.title('proposal_level_1')
                plt.subplot(235)
                plt.imshow(overlay_img_with_proposals(input_img, interp_proposal_masks[2][view_index]))
                # plt.imshow(per_level_proposal_masks[2][view_index], vmin=0.0, vmax=1.0)
                plt.axis("off")
                plt.title('proposal_level_2')
                plt.subplot(236)
                plt.imshow(overlay_img_with_proposals(input_img, interp_proposal_masks[3][view_index]))
                # plt.imshow(per_level_proposal_masks[3][view_index], vmin=0.0, vmax=1.0)
                plt.axis("off")
                plt.title('proposal_level_3')
                plt.savefig(
                    "{}/view_{}".format(save_path, cam_index),
                )
                plt.close(0)
        