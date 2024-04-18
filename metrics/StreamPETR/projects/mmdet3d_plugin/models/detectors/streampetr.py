# ---------------------------------------------
#  Modified by Yuqing Wen
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.misc import locations
from fairscale.nn.checkpoint import checkpoint_wrapper
from mmdet.core import bbox2result
from mmdet.apis import show_result_pyplot
import mmcv
from mmdet.core.visualization import imshow_det_bboxes
import os
import numpy as np

@DETECTORS.register_module()
class StreamPETR(MVXTwoStageDetector):
    """StreamPETR."""

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 stride=[16],
                 position_level=[0],
                 aux_2d_only=True,
                 query_2d=False,
                 show_2d=False,
                 loss_lamda=1.0,
                 single_test=False,
                 pretrained=None):
        super(StreamPETR, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.prev_scene_token = None
        self.single_test = single_test
        self.stride = stride
        self.position_level = position_level
        self.aux_2d_only = aux_2d_only
        self.query_2d = query_2d
        self.show_2d = show_2d
        self.loss_lamda = loss_lamda
        # self.img_roi_head = checkpoint_wrapper(img_roi_head)

    def extract_img_feat(self, img):
        """Extract features of images."""
        B = img.size(0)

        if img is not None:
            if img.dim() == 6:
                img = img.flatten(1, 2)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for i in self.position_level:
            BN, C, H, W = img_feats[i].size()
            img_feat_reshaped = img_feats[i].view(B, int(BN/B), C, H, W)
            img_feats_reshaped.append(img_feat_reshaped)

        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, training_mode=False):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img)
        return img_feats

    def prepare_location(self, img_metas, **data):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        assert len(self.stride) == len(data['img_feats'])
        location_r = []
        for i in range(len(data['img_feats'])):
            bs, n = data['img_feats'][i].shape[:2]
            x = data['img_feats'][i].flatten(0, 1)
            location = locations(x, self.stride[i], pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
            location_r.append(location)
        return location_r

    def forward_roi_head(self, location, **data):
        # if (self.aux_2d_only and not self.query_2d) or not self.with_img_roi_head:
        #     return {'topk_indexes':None}
        # else:
        #     outs_roi = self.img_roi_head(data['img_feats'])
        #     return outs_roi
        
        outs_roi = self.img_roi_head(data['img_feats'])
        return outs_roi


    def forward_pts_train(self,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_bboxes,
                          gt_labels,
                          img_metas,
                          centers2d,
                          depths,
                          **data):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
            proposals_list: 12, (torch.Size([100, 5]) torch.Size([100]))
        """
        location = self.prepare_location(img_metas, **data)
        if self.with_img_roi_head:
            outs_roi = self.forward_roi_head(location, **data)
            if self.query_2d:
                with torch.no_grad():
                    proposals_list = self.img_roi_head.get_bboxes(outs_roi, img_metas, rescale=False)
                    # bbox_results = [
                    #     bbox2result(det_bboxes, det_labels, self.img_roi_head.num_classes)
                    #     for det_bboxes, det_labels in results_list
                    # ]
                    data["proposal"] = proposals_list

        # if self.pts_bbox_head.type == "JointStreamPETRHead":

        # topk_indexes = outs_roi['topk_indexes']
        # # data['img_feats'] = [[], data['img_feats'][1]]
        # location = location[1]
        # outs = self.pts_bbox_head(location, img_metas, topk_indexes, **data)

        outs = self.pts_bbox_head(img_metas, **data)

        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        for key, value in losses.items():
            losses[key] = value * self.loss_lamda

        if self.with_img_roi_head:
            loss2d_inputs = [gt_bboxes, gt_labels, centers2d, depths, outs_roi, img_metas]
            losses2d = self.img_roi_head.loss(*loss2d_inputs)
            for key, value in losses2d.items():
                # if isinstance(value, list):
                #     value = [torch.nan_to_num(v) for v in value]
                # else:
                #     value = torch.nan_to_num(value)
                losses[key.replace("loss", "loss2d")] = value 

        return losses

    @force_fp32(apply_to=('img'))
    def forward(self, return_loss=True, **data):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**data)
        else:
            return self.forward_test(**data)

    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      depths=None,
                      centers2d=None,
                      **data):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        for key in data:
            data[key] = data[key][:, 0] 
        rec_img = data['img']
        rec_img_feats = self.extract_feat(rec_img)

        data['img_feats'] = rec_img_feats

        losses = self.forward_pts_train(gt_bboxes_3d,
                                    gt_labels_3d, gt_bboxes,
                                    gt_labels, img_metas, centers2d, depths, **data)

        return losses
  
  
    def forward_test(self, img_metas, rescale, **data):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        for key in data:
            if key != 'img':
                data[key] = data[key][0][0].unsqueeze(0)
            else:
                data[key] = data[key][0]
        return self.simple_test(img_metas[0], **data)

    def simple_test_pts(self, img_metas, **data):
        """Test function of point cloud branch."""
        location = self.prepare_location(img_metas, **data)
        outs_roi = self.forward_roi_head(location, **data)
        topk_indexes = outs_roi['topk_indexes']

        if img_metas[0]['scene_token'] != self.prev_scene_token:
            self.prev_scene_token = img_metas[0]['scene_token']
            data['prev_exists'] = data['img'].new_zeros(1)
            self.pts_bbox_head.reset_memory()
        else:
            data['prev_exists'] = data['img'].new_ones(1)

        bbox2d_results = None
        if self.query_2d:
            proposals_list = self.img_roi_head.get_bboxes(outs_roi, img_metas, rescale=False)
            data["proposal"] = proposals_list
            # proposals_list = self.img_roi_head.get_bboxes(outs_roi, img_metas, rescale=True)
            bbox2d_results = [
                bbox2result(det_bboxes, det_labels, self.img_roi_head.num_classes)
                for det_bboxes, det_labels in proposals_list
            ]

        outs = self.pts_bbox_head(img_metas, **data)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        if bbox2d_results is not None:
            for img_id in range(len(bbox_results)):
                bbox3d = bbox_results[img_id]
                bbox3d['bbox2d'] = bbox2d_results
                bbox_results[img_id] = bbox3d
        
        if self.show_2d:
            self.show2d_results(bbox_results, img_metas, data['img'])

        return bbox_results
    
    def simple_test(self, img_metas, **data):
        """Test function without augmentaiton."""
        data['img_feats'] = self.extract_img_feat(data['img'])

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            img_metas, **data)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def show2d_results(self,
                    bbox_results, 
                    img_metas,
                    imgs,
                    score_thr=0.1,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=1,
                    font_size=7,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file="./demo/"):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        filenames = [filename for img_meta in img_metas for filename in img_meta['filename']]
        bbox2d_results = [bbox2d for bbox_result in bbox_results for bbox2d in bbox_result['bbox2d']]
        mean = torch.from_numpy(img_metas[0]['img_norm_cfg']['mean']).view(1, -1, 1, 1).to(imgs.device)
        std = torch.from_numpy(img_metas[0]['img_norm_cfg']['std']).view(1, -1, 1, 1).to(imgs.device)
        imgs = imgs.clone()*std + mean
        imgs = imgs.permute((0, 2, 3, 1)).detach().cpu().numpy()
        for img_id in range(len(filenames)):
            # img = mmcv.imread(filenames[img_id])
            result = bbox2d_results[img_id]
            img = imgs[img_id].copy()
            if isinstance(result, tuple):
                bbox_result, segm_result = result
                if isinstance(segm_result, tuple):
                    segm_result = segm_result[0]  # ms rcnn
            else:
                bbox_result, segm_result = result, None
            bboxes = np.vstack(bbox_result)
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)
            # draw segmentation masks
            segms = None
            if segm_result is not None and len(labels) > 0:  # non empty
                segms = mmcv.concat_list(segm_result)
                if isinstance(segms[0], torch.Tensor):
                    segms = torch.stack(segms, dim=0).detach().cpu().numpy()
                else:
                    segms = np.stack(segms, axis=0)
            # if out_file specified, do not show image in window
            if out_file is not None:
                show = False
            # draw bounding boxes
            img = imshow_det_bboxes(
                img,
                bboxes,
                labels,
                segms,
                class_names=self.CLASSES,
                score_thr=score_thr,
                bbox_color=bbox_color,
                text_color=text_color,
                mask_color=mask_color,
                thickness=thickness,
                font_size=font_size,
                win_name=win_name,
                show=show,
                wait_time=wait_time,
                out_file=None)
            # print(img.shape)
            mmcv.imwrite(img, os.path.join(out_file, filenames[img_id].split("/")[-1]))
            if not (show or out_file):
                return img



    