import torch
import mmcv
import numpy as np
import time
import os

from os import path as osp
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmcv.parallel import DataContainer as DC
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from mmdet3d.core import (
    LiDARInstance3DBoxes,
    CameraInstance3DBoxes,
    Box3DMode,
    bbox3d2result,
    show_multi_modality_result,
)

from mmdet3d.core import Box3DMode, Coord3DMode, show_result
import pdb

@DETECTORS.register_module()
class SimMOD(MVXTwoStageDetector):
    def __init__(
        self,
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
        pretrained=None,
        filter_gt_with_proposals=True,
    ):
        super(SimMOD, self).__init__(
            pts_voxel_layer,
            pts_voxel_encoder,
            pts_middle_encoder,
            pts_fusion_layer,
            img_backbone,
            pts_backbone,
            img_neck,
            pts_neck,
            pts_bbox_head,
            img_roi_head,
            img_rpn_head,
            train_cfg,
            test_cfg,
            pretrained,
        )

        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
        )
        self.use_grid_mask = use_grid_mask

        # filter ground-truth with generated proposals
        self.filter_gt_with_proposals = filter_gt_with_proposals

        # loss weights for two stages
        if self.train_cfg and "two_stage_loss_weights" in self.train_cfg:
            self.two_stage_loss_weights = self.train_cfg["two_stage_loss_weights"]
        else:
            self.two_stage_loss_weights = [1.0, 1.0]
        
        if img_roi_head is None:
            self.img_roi_head = None

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None

        # with swin_transformer: [torch.Size([6, 96, 232, 400]), torch.Size([6, 192, 116, 200]), torch.Size([6, 384, 58, 100]), torch.Size([6, 768, 29, 50])]
        # with res50 / res101: [torch.Size([6, 256, 232, 400]), torch.Size([6, 512, 116, 200]), torch.Size([6, 1024, 58, 100]), torch.Size([6, 2048, 29, 50])]

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped

    @auto_fp16(apply_to=("img"), out_fp32=True)
    def extract_feat(self, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)

        return img_feats

    def forward_pts_train(
        self,
        pts_feats,
        gt_bboxes_3d,
        gt_labels_3d,
        img_metas,
        gt_bboxes_ignore=None,
        proposal_features=None,
        proposal_pos_embeddings=None,
        proposal_scores=None,
        proposal_match_idxs=None,
        valid_ranges=None,
        raw_img=None,
    ):
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
        """

        outs = self.pts_bbox_head(
            mlvl_feats=pts_feats,
            img_metas=img_metas,
            proposal_features=proposal_features,
            proposal_positions=proposal_pos_embeddings,
            proposal_scores=proposal_scores,
            valid_ranges=valid_ranges,
            raw_img=raw_img,
        )
        
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, preds_match_idxs=proposal_match_idxs)

        return losses

    @force_fp32(apply_to=("img", "points"))
    def forward(self, return_loss=True, **kwargs):
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
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(
        self,
        points=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_labels=None,
        gt_bboxes=None,
        img=None,
        proposals=None,
        gt_bboxes_ignore=None,
        img_depth=None,
        img_mask=None,
        cam_anno_infos=None,
        raw_img=None,
        valid_ranges=None,
    ):
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

        """
        backbone + fpn ==> multi-stage features in p3, p4, p5, p6
        for img_input of size (928, 1600), the sizes of feature maps are [torch.Size([1, 6, 256, 116, 200]), torch.Size([1, 6, 256, 58, 100]), torch.Size([1, 6, 256, 29, 50]), torch.Size([1, 6, 256, 15, 25])]
        """
        # print(len(cam_anno_infos))
        # print(len(cam_anno_infos[0]))
        cam_anno_infos = cam_anno_infos[0]
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        losses = dict()
        """
        fcos_head ==> generating proposals (and one-stage losses in training)
        """

        if self.img_roi_head is None:
            if self.two_stage_loss_weights[1] > 0:
                losses_pts = self.forward_pts_train(
                    pts_feats=img_feats,
                    gt_bboxes_3d=gt_bboxes_3d,
                    gt_labels_3d=gt_labels_3d,
                    img_metas=img_metas,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    valid_ranges=valid_ranges,
                    raw_img=raw_img,
                )

                for key, val in losses_pts.items():
                    losses[key] = val * self.two_stage_loss_weights[1]
        
        else:
            fcos_outputs = self.img_roi_head(img_feats)
            # print(fcos_outputs)
            fcos_losses, fcos_utils = self.img_roi_head.get_loss(
                fcos_outputs,
                cam_anno_infos,
                img_metas=img_metas,
                global_gt_bboxes_3d=gt_bboxes_3d,
                raw_img=raw_img,
            )

            # scale the losses for the first stage
            for key, val in fcos_losses.items():
                losses["fcos_loss_{}".format(key)] = val * self.two_stage_loss_weights[0]

            res_proposals = self.img_roi_head.get_proposals(fcos_outputs, fcos_utils,
                img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d, raw_img=raw_img)

            """
            detr_head ==> the merge & refinement in the second stage
            """
            recall = len(torch.unique(res_proposals['proposal_match_idxs'])) / gt_bboxes_3d[0].tensor.shape[0]
            losses['proposal_recall'] = torch.tensor(recall).cuda()

            with torch.no_grad():
                if self.filter_gt_with_proposals:
                    proposal_filter_indices = res_proposals['proposal_match_idxs']
                    gt_bboxes_3d = [gt_bboxes_3d[0][proposal_filter_indices]]
                    gt_labels_3d = [gt_labels_3d[0][proposal_filter_indices]]

            if self.two_stage_loss_weights[1] > 0:
                losses_pts = self.forward_pts_train(
                    pts_feats=img_feats,
                    gt_bboxes_3d=gt_bboxes_3d,
                    gt_labels_3d=gt_labels_3d,
                    img_metas=img_metas,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_features=res_proposals['proposal_features'],
                    proposal_pos_embeddings=res_proposals['proposal_positions'],
                    proposal_scores=res_proposals['proposal_scores'],
                    proposal_match_idxs=res_proposals['proposal_corr_idxs'],
                    valid_ranges=valid_ranges,
                    raw_img=raw_img,
                )

                for key, val in losses_pts.items():
                    losses[key] = val * self.two_stage_loss_weights[1]
        
        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))

        img = [img] if img is None else img
        return self.simple_test(img_metas[0], img[0], **kwargs)

    def simple_test_pts(
        self,
        x,
        img_metas,
        proposal=None,
        proposal_pos=None,
        rescale=False,
        proposal_scores=None,
        proposal_uncertainties=None,
        raw_img=None,
    ):
        """Test function of point cloud branch."""

        outs = self.pts_bbox_head(x, img_metas, proposal, proposal_pos,
                proposal_scores=proposal_scores, raw_img=raw_img)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        return bbox_results

    def simple_test(self, img_metas, img=None, rescale=False, raw_img=None):
        """Test function without augmentaiton."""

        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        # tuple of (cls_score, bbox_pred, dir_cls_pred, attr_pred, centerness, cls_feat, reg_feat)
        # each element contains multi-level predictions
        fcos_outputs = self.img_roi_head(img_feats)
        bbox_list = [dict() for i in range(len(img_metas))]

        res_proposals = self.img_roi_head.get_proposals(
            fcos_outputs=fcos_outputs,
            img_metas=img_metas,
        )

        # list of dic with keys = ['boxes_3d', 'scores_3d', 'labels_3d']
        bbox_pts = self.simple_test_pts(
            img_feats,
            img_metas,
            proposal=res_proposals['proposal_features'],
            proposal_pos=res_proposals['proposal_positions'],
            proposal_scores=res_proposals['proposal_scores'],
            rescale=rescale,
            raw_img=raw_img,
        )

        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox

        return bbox_list

    def aug_test_pts(self, feats, img_metas, rescale=False):
        feats_list = []
        for j in range(len(feats[0])):
            feats_list_level = []
            for i in range(len(feats)):
                feats_list_level.append(feats[i][j])
            feats_list.append(torch.stack(feats_list_level, -1).mean(-1))
        outs = self.pts_bbox_head(feats_list, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats = self.extract_feats(img_metas, imgs)
        img_metas = img_metas[0]
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.aug_test_pts(img_feats, img_metas, rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox

        return bbox_list

    def show_results(self, data, result, out_dir):
        """Results visualization.

        Args:
            data (list[dict]): Input images and the information of the sample.
            result (list[dict]): Prediction results.
            out_dir (str): Output directory of visualization result.
        """

        from mmdet3d.core.visualizer.image_vis import (
            draw_camera_bbox3d_on_img,
            draw_depth_bbox3d_on_img,
            draw_lidar_bbox3d_on_img,
        )

        camera_types = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]

        import os

        visualize_thresh = 0.2
        for batch_id in range(len(result)):
            img_metas_data = data["img_metas"][0]._data[0][batch_id]
            img_filenames = img_metas_data["filename"]
            # lidar to cam
            lidar2cams = img_metas_data["lidar2cam"]
            # cam to img
            cam2imgs = img_metas_data["cam2img"]
            # lidar to img
            lidar2imgs = img_metas_data["lidar2img"]

            # ground-truth objects
            gt_lidar_boxes = data['gt_bboxes_3d'][0]._data[0][batch_id]
            gt_labels = data['gt_labels_3d'][0]._data[0][batch_id]
            
            # predictions
            bbox_results = result[batch_id]["pts_bbox"]
            pred_lidar_boxes = bbox_results["boxes_3d"]
            pred_labels = bbox_results['labels_3d']

            pred_scores_3d = bbox_results["scores_3d"]
            valid_mask = pred_scores_3d > visualize_thresh
            
            pred_lidar_boxes = pred_lidar_boxes[valid_mask]
            pred_labels = pred_labels[valid_mask]

            if gt_lidar_boxes.shape[0] < 20:
                continue
        
            gt_imgs = {}
            pred_imgs = {}

            gt_bbox_color = (61, 102, 255)
            pred_bbox_color = (241, 101, 72)

            for cam_type, img_filename, lidar2img in zip(
                camera_types, img_filenames, lidar2imgs
            ):
                img = mmcv.imread(img_filename)
                file_name = osp.split(img_filename)[-1].split(".")[0]
                assert out_dir is not None, "Expect out_dir, got none."

                img_with_gt = draw_lidar_bbox3d_on_img(gt_lidar_boxes, img, lidar2img, None, color=gt_bbox_color)

                img_with_pred = draw_lidar_bbox3d_on_img(pred_lidar_boxes, img, lidar2img, None, color=pred_bbox_color)

                mmcv.imwrite(img, osp.join(out_dir, "{}_img.png".format(cam_type)))
                mmcv.imwrite(img_with_gt, osp.join(out_dir, "{}_gt.png".format(cam_type)))
                mmcv.imwrite(img_with_pred, osp.join(out_dir, "{}_pred.png".format(cam_type)))

                gt_imgs[cam_type] = img_with_gt
                pred_imgs[cam_type] = img_with_pred
            
            # visualize LiDAR bounding boxes
            # points = data['points']._data[0][batch_id]
            # points = Coord3DMode.convert_point(
            #         points, Coord3DMode.LIDAR, Coord3DMode.DEPTH
            #     )
            # box_mode_3d = Box3DMode.LIDAR
            # gt_depth_bboxes = Box3DMode.convert(
            #     gt_lidar_boxes, box_mode_3d, Box3DMode.DEPTH
            # )
            # pred_depth_bboxes = Box3DMode.convert(
            #     pred_lidar_boxes, box_mode_3d, Box3DMode.DEPTH
            # )
            # gt_depth_bboxes = gt_depth_bboxes.tensor.cpu().numpy()
            # pred_depth_bboxes = pred_depth_bboxes.tensor.cpu().numpy()

            # show_result(points, gt_depth_bboxes, pred_depth_bboxes, out_dir, file_name)

            # draw grid of images
            import matplotlib as mpl
            import matplotlib.pyplot as plt
            import os
            
            val_w = 6.4
            val_h = val_w / 16 * 9
            fig = plt.figure(figsize=(3 * val_w, 4 * val_h))
            width_ratios = (val_w, val_w, val_w)
            gs = mpl.gridspec.GridSpec(4, 3, width_ratios=width_ratios)
            # gs.update(wspace=0.01, hspace=0.1, left=0,
            #           right=1.0, top=1.0, bottom=0.1)
            
            vis_orders = [
                "CAM_FRONT_LEFT",
                "CAM_FRONT",
                "CAM_FRONT_RIGHT",
                "CAM_BACK_LEFT",
                "CAM_BACK",
                "CAM_BACK_RIGHT",
            ]

            label_font_size = 16

            for img_index, vis_cam_type in enumerate(vis_orders):
                vis_gt_img = gt_imgs[vis_cam_type]
                vis_pred_img = pred_imgs[vis_cam_type]

                # prediction
                ax = plt.subplot(gs[(img_index // 3) * 2, img_index % 3])
                # plt.annotate(vis_cam_type.replace('_', ' ').replace(
                #     'CAM ', ''), (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
                # plt.annotate(vis_cam_type.replace('_', ' ').replace(
                #     'CAM ', ''), (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
                plt.imshow(vis_pred_img)
                # if img_index % 3 == 0:
                plt.title(vis_cam_type, fontsize=label_font_size)
                plt.axis('off')
                ax.set_ylabel("Prediction", fontsize=label_font_size)
                plt.draw()

                # ground-truth
                ax = plt.subplot(gs[(img_index // 3) * 2 + 1, img_index % 3])
                # if img_index % 3 == 0:
                plt.imshow(vis_gt_img)
                plt.axis('off')
                ax.set_ylabel("Ground-truth", fontsize=label_font_size)
                plt.draw()
            
            plt.tight_layout()
            plt.savefig(out_dir + '_vis.png')
            # plt.savefig(os.path.join(out_dir, 'global_vis.pdf'))
            plt.close()
            
            pdb.set_trace()