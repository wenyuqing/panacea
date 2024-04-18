import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION, TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import (
    MultiScaleDeformableAttention,
    TransformerLayerSequence,
    build_transformer_layer_sequence,
)
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn import Linear, bias_init_with_prob
from torch.nn.init import normal_

import pdb


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    
    return torch.log(x1 / x2)

@TRANSFORMER.register_module()
class TwoStageDetr3DTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(
        self,
        num_feature_levels=4,
        num_cams=6,
        two_stage_num_proposals=300,
        decoder=None,
        point_cloud_range=None,
        **kwargs,
    ):
        super(TwoStageDetr3DTransformer, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.two_stage_num_proposals = two_stage_num_proposals
        self.point_cloud_range = point_cloud_range
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        pass

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention) or isinstance(
                m, TwoStageDetr3DCrossAtten
            ):
                m.init_weight()

    def forward(
        self,
        mlvl_feats,
        query,
        query_pos=None,
        reg_branches=None,
        query_reference_points=None,
        **kwargs,
    ):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        bs = mlvl_feats[0].size(0)

        reference_points = query_reference_points
        init_reference_out = reference_points.clone()

        # decoder
        query = query.permute(1, 0, 2)
        if query_pos is not None:
            query_pos = query_pos.permute(1, 0, 2)

        inter_states, inter_references, inter_outputs = self.decoder(
            query=query,
            key=None,
            value=mlvl_feats,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            **kwargs,
        )

        return inter_states, init_reference_out, inter_references, inter_outputs


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class TwoStageDetr3DTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, embed_cam_level_encoding=False,
            iterative_pos_encoding=False, output_layer=None, **kwargs):
        super(TwoStageDetr3DTransformerDecoder, self).__init__(*args, **kwargs)

        self.return_intermediate = return_intermediate
        self.output_layer = output_layer if output_layer is not None else self.num_layers - 1
        
        # generate pos_encodings within each iteration
        self.iterative_pos_encoding = iterative_pos_encoding
        if self.iterative_pos_encoding:
            # project (x, y, z) information
            self.pos_proj = nn.Linear(3, self.embed_dims)
            
            # embed camera & level information into the encodings
            self.embed_cam_level_encoding = embed_cam_level_encoding
            if self.embed_cam_level_encoding:
                self.proposal_level_embeds = nn.Embedding(4, self.embed_dims // 2)
                self.proposal_cam_embeds = nn.Embedding(6, self.embed_dims // 2)
                self.pos_reduce = Linear(self.embed_dims * 2, self.embed_dims)

    def forward(self, query, query_pos, *args, reference_points=None, reg_branches=None, **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        intermediate_output = []

        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points
            
            if self.iterative_pos_encoding:
                query_pos = self.pos_proj(reference_points_input)

                if self.embed_cam_level_encoding:
                    query_level_encoding = self.proposal_level_embeds.weight[kwargs['proposal_levels'].long()]
                    query_cam_encoding = self.proposal_cam_embeds.weight[kwargs['proposal_views'].long()]
                    query_pos = torch.cat((query_pos, query_level_encoding, query_cam_encoding), dim=2)
                    query_pos = self.pos_reduce(query_pos)

                # [B, N, C] ==> [N, B, C]
                query_pos = query_pos.permute(1, 0, 2)
            
            # [N, B, C] as input
            last_layer = (lid == len(self.layers) - 1)
            output = layer(
                output, *args, query_pos=query_pos, reference_points=reference_points_input, last_layer=last_layer, **kwargs
            )

            # [N, B, C] ==> [B, N, C]
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)

                if self.return_intermediate:
                    intermediate_output.append(tmp)

                assert reference_points.shape[-1] == 3
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(
                    reference_points[..., :2]
                )
                new_reference_points[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(
                    reference_points[..., 2:3]
                )
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return (
                torch.stack(intermediate),
                torch.stack(intermediate_reference_points),
                torch.stack(intermediate_output),
            )

        return output, reference_points


@ATTENTION.register_module()
class TwoStageDetr3DCrossAtten(BaseModule):
    """An attention module used in Detr3d.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=5,
        num_cams=6,
        im2col_step=64,
        pc_range=None,
        dropout=0.1,
        norm_cfg=None,
        init_cfg=None,
        batch_first=False,
        attention_weights=True,
        attention_weights_normalize=False,
        pos_encoder_linear=False,
        depth_wise_weights=False,
        debug=False,
    ):
        super(TwoStageDetr3DCrossAtten, self).__init__(init_cfg)

        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, "
                f"but got {embed_dims} and {num_heads}"
            )

        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n))
                )
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                "MultiScaleDeformAttention to make "
                "the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams

        self.using_attention_weights = attention_weights
        self.depth_wise_weights = depth_wise_weights
        if self.using_attention_weights:
            if self.depth_wise_weights:
                self.attention_weights = nn.Linear(
                    embed_dims, num_cams + num_levels + num_points)
            else:
                self.attention_weights = nn.Linear(
                    embed_dims, num_cams * num_levels * num_points
                )

        # normalize the aggregated features from multi-view and multi-scales
        self.attention_weights_normalize = attention_weights_normalize

        # embed location into information
        self.pos_encoder_linear = pos_encoder_linear
        if self.pos_encoder_linear:
            self.position_encoder = nn.Linear(3, self.embed_dims)
        else:
            self.position_encoder = nn.Sequential(
                nn.Linear(3, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
            )

        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.init_weight()

        self.debug = debug

    def init_weight(self):
        """Default initialization for Parameters of Module."""

        if hasattr(self, "attention_weights"):
            constant_init(self.attention_weights, val=0.0, bias=0.0)

        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        query,
        key,
        value,
        residual=None,
        query_pos=None,
        key_padding_mask=None,
        zz=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ranges=None,
        raw_img=None,
        last_layer=False,
        **kwargs,
    ):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query

        if value is None:
            value = key

        if residual is None:
            inp_residual = query

        # the query_pos comes from the learnable parameters and stays the same for each iteration through the decoding phase
        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)
        bs, num_query, _ = query.size()

        # normalized_reference_points_2d, [batch, num_cam, num_query, 2]
        reference_points_3d, norm_pts_2d, output, mask = feature_sampling(value, 
            reference_points, self.pc_range, kwargs["img_metas"], valid_ranges=valid_ranges)
        
        """
        img_metas include dict_keys(['filename', 'ori_shape', 'img_shape', 'lidar2img', 'pad_shape', 
                'scale_factor', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'sample_idx', 'pts_filename', 'input_shape'])
        """

        if last_layer and self.debug:
            debug_visualize_reference_points_2d(
                norm_pts_2d, mask, raw_img, kwargs["img_metas"], color="lime",
            )

        if self.depth_wise_weights:
            camera_weights, point_weights, level_weights = torch.split(self.attention_weights(query), [self.num_cams, self.num_points, self.num_levels], dim=2)
            attention_weights = camera_weights.view(bs, num_query, -1, 1, 1) * point_weights.view(bs, num_query, 1, -1, 1) * level_weights.view(bs, num_query, 1, 1, -1)
            attention_weights = attention_weights.view(bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)
        else:
            attention_weights = self.attention_weights(query).view(
                bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)
 
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)

        attention_weights = attention_weights.sigmoid() * mask
        output = output * attention_weights
        output = output.sum(-1).sum(-1).sum(-1)
        output = output.permute(2, 0, 1)
        
        output = self.output_proj(output)
        # (num_query, bs, embed_dims)
        pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)

        return self.dropout(output) + inp_residual + pos_feat

def debug_visualize_reference_points_2d(points_2d, points_mask, raw_imgs, img_metas, 
        color='r', save_folder='simmod'):
    
    """
    normalized_reference_points_2d: tensor of shape (batch, num_cam, num_query, 2)
    """
    import os
    import imageio
    import matplotlib.pyplot as plt

    batch_size, num_cam, num_query = points_2d.shape[:-1]
    save_path = os.path.join("debugs/sampling_pts", save_folder)
    os.makedirs(save_path, exist_ok=True)

    points_2d = (points_2d + 1) / 2
    points_2d = points_2d.detach().cpu().numpy()

    points_mask = points_mask.squeeze(1).squeeze(-1)
    points_mask = points_mask.permute(0, 2, 1, 3).flatten(2).detach().cpu().numpy()

    if type(raw_imgs) is list:
        raw_imgs = raw_imgs[0]
    
    visualize_order = [2, 0, 1, 4, 3, 5]

    for batch_index in range(batch_size):
        num_row, num_col = 2, 3

        # [Front, Front_right, FRONT_left]
        # [BACK, BACK_LEFT, BACK_RIGHT]

        plt.figure(0, figsize=(num_col * 16, num_row * 9))
        for cam_index in range(num_cam):
            vis_index = visualize_order[cam_index]

            raw_img = raw_imgs[batch_index][vis_index].detach().cpu().numpy().astype(np.uint8)
            # bgr to rgb
            raw_img = raw_img[..., [2, 1, 0]]
            img_shape = img_metas[batch_index]["img_shape"][vis_index]
            
            per_points_2d = points_2d[batch_index, vis_index]
            per_mask = points_mask[batch_index, vis_index]

            per_points_2d = per_points_2d[per_mask.astype(np.bool)]
            per_points_2d[:, 0] *= img_shape[1]
            per_points_2d[:, 1] *= img_shape[0]
            per_points_2d = per_points_2d.astype(np.int)

            plt.subplot(num_row, num_col, cam_index + 1)
            plt.imshow(raw_img)
            plt.scatter(
                per_points_2d[:, 0],
                per_points_2d[:, 1],
                c=color,
                s=192,
                marker='*',
                alpha=0.8,
            )
            plt.axis("off")
        
        sample_token = img_metas[batch_index]['sample_idx']
        plt.tight_layout()
        plt.savefig(
            "{}/{}.png".format(save_path, sample_token),
        )
        plt.close(0)
    
def feature_sampling(mlvl_feats, reference_points, pc_range, img_metas, valid_ranges=None):
    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta["lidar2img"])
    
    img_shape = img_metas[0]["img_shape"][0]

    lidar2img = np.asarray(lidar2img)
    lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
    reference_points = reference_points.clone()
    reference_points_3d = reference_points.clone()
    reference_points[..., 0:1] = (
        reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    )
    reference_points[..., 1:2] = (
        reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    )
    reference_points[..., 2:3] = (
        reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
    )

    reference_points = torch.cat(
        (reference_points, torch.ones_like(reference_points[..., :1])), -1
    )

    B, num_query = reference_points.size()[:2]
    num_cam = lidar2img.size(1)
    reference_points = (
        reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
    )
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
    eps = 1e-5
    mask = reference_points_cam[..., 2:3] > eps
    reference_points_cam = reference_points_cam[..., 0:2] / torch.clamp_min(reference_points_cam[..., 2:3], eps)

    reference_points_cam[..., 0] /= img_shape[1]
    reference_points_cam[..., 1] /= img_shape[0]
    reference_points_cam = (reference_points_cam - 0.5) * 2
    normalized_reference_points_2d = reference_points_cam.clone()

    if valid_ranges is None:
        batch, num_cam = reference_points_cam.shape[:2]
        valid_ranges = torch.tensor([0, 0, img_shape[1], img_shape[0]]).view(1, 1, -1).type_as(reference_points_cam)
    
    # normalize ranges   
    normalizer = torch.tensor([img_shape[1], img_shape[0], img_shape[1], img_shape[0]])
    normalize_valid_ranges = valid_ranges / normalizer.type_as(valid_ranges).float()
    normalize_valid_ranges = (normalize_valid_ranges - 0.5) * 2

    batch, num_cam, num_proposal = reference_points_cam.shape[:3]
    normalize_valid_ranges = normalize_valid_ranges.unsqueeze(2).expand(batch, num_cam, num_proposal, -1)
    
    mask = (
        mask
        & (reference_points_cam[..., 0:1] > normalize_valid_ranges[..., 0:1])
        & (reference_points_cam[..., 0:1] < normalize_valid_ranges[..., 2:3])
        & (reference_points_cam[..., 1:2] > normalize_valid_ranges[..., 1:2])
        & (reference_points_cam[..., 1:2] < normalize_valid_ranges[..., 3:])
    )

    mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
    mask = torch.nan_to_num(mask)

    # sampling features
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        feat = feat.view(B * N, C, H, W)
        reference_points_cam_lvl = reference_points_cam.view(B * N, num_query, 1, 2)
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl, align_corners=True)
        sampled_feat = sampled_feat.view(B, N, C, num_query, 1).permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)

    sampled_feats = torch.stack(sampled_feats, -1)
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam, 1, len(mlvl_feats))

    return reference_points_3d, normalized_reference_points_2d, sampled_feats, mask

    