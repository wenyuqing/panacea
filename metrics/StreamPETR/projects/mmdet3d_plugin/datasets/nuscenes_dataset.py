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

import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from mmdet.datasets import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import Quaternion
from mmcv.parallel import DataContainer as DC
import random
import math
import mmcv
import cv2
from os import path as osp
from PIL import Image
from mmdet3d.datasets.nuscenes_dataset import output_to_nusc_box, lidar_nusc_box_to_global

@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, collect_keys, seq_mode=False, seq_split_num=1, num_frame_losses=1, queue_length=8, random_length=0, sliding_window=False, root_path=None, filter_file=None, sample_rate=None, *args, **kwargs):
        self.filter_file = filter_file
        self.sample_rate = sample_rate
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.collect_keys = collect_keys
        self.random_length = random_length
        self.num_frame_losses = num_frame_losses
        self.seq_mode = seq_mode
        self.sliding_window = sliding_window
        self.root_path = root_path
        

        if seq_mode:
            self.num_frame_losses = 1
            self.queue_length = 1
            self.seq_split_num = seq_split_num
            self.random_length = 0
            self._set_sequence_group_flag() # Must be called after load_annotations b/c load_annotations does sorting.
        # set group flag for the samplers
        if not self.test_mode and filter_file is not None:
            self._reset_group_flag()
            
    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        if self.filter_ids is None:
            return len(self.data_infos)
        else:
            return len(self.filter_ids)

    def _reset_group_flag(self):
        """Set flag according to image aspect ratio.
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)


    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file, file_format='pkl')
        # print(len(data['infos']))
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        if self.sample_rate is not None:
            num_sample = int(len(data_infos) * self.sample_rate)
            data_infos = data_infos[:num_sample]
        # data_infos = data_infos[:200]

        if self.filter_file is not None:
            filter_data = mmcv.load(self.filter_file, file_format='pkl')
            filter_infos = list(sorted(filter_data['infos'], key=lambda e: e['timestamp']))
            self.filter_tokens = [filter_info['token'] for filter_info in filter_infos]

            self.filter_ids = [i for i in range(len(data_infos)) if data_infos[i]['token'] in self.filter_tokens]
            self.index_map = {}
            for i, index in enumerate(self.filter_ids):
                self.index_map[i] = index
        else:
            self.filter_tokens = None
            self.filter_ids = None
            self.index_map = None

        return data_infos

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []

        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and len(self.data_infos[idx]['sweeps']) == 0:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.seq_split_num != 1:
            if self.seq_split_num == 'all':
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0, 
                                bin_counts[curr_flag], 
                                math.ceil(bin_counts[curr_flag] / self.seq_split_num)))
                        + [bin_counts[curr_flag]])

                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.seq_split_num
                self.flag = np.array(new_flags, dtype=np.int64)


    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []
        index_list = list(range(index-self.queue_length-self.random_length+1, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[self.random_length:])
        index_list.append(index)
        prev_scene_token = None
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            
            if not self.seq_mode: # for sliding window only
                if input_dict['scene_token'] != prev_scene_token:
                    input_dict.update(dict(prev_exists=False))
                    prev_scene_token = input_dict['scene_token']
                else:
                    input_dict.update(dict(prev_exists=True))

            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)

            queue.append(example)

        for k in range(self.num_frame_losses):
            if self.filter_empty_gt and \
                (queue[-k-1] is None or ~(queue[-k-1]['gt_labels_3d']._data != -1).any()):
                return None
        return self.union2one(queue)

    def prepare_pseudotrain_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []
        index_list = list(range(index-self.queue_length-self.random_length+1, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[self.random_length:])
        index_list.append(index)

        token = self.data_infos[index]['token']
        if self.filter_tokens is not None and token not in self.filter_tokens:
            return None

        if self.modality['use_camera']:
            image_paths = []
            for cam_type, cam_info in self.data_infos[index]['cams'].items():
                image_paths.append(cam_info['data_path'])

        prev_scene_token = None
        for s_index, i in enumerate(index_list):
            i = max(0, i)
            input_dict = self.get_data_info(i)

            img_list = []
            for filename in image_paths:
                img_path = self.root_path + filename.split("__")[1] + "_" + filename.split("/")[-1].replace(".jpg", "/") +"_00000"+ str(s_index)+".jpg"
                img_list.append(img_path)
            input_dict['img_filename']=img_list

            input_dict['pseudo'] = True
            
            if not self.seq_mode: # for sliding window only
                if input_dict['scene_token'] != prev_scene_token:
                    input_dict.update(dict(prev_exists=False))
                    prev_scene_token = input_dict['scene_token']
                else:
                    input_dict.update(dict(prev_exists=True))

            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)

            queue.append(example)

        for k in range(self.num_frame_losses):
            if self.filter_empty_gt and \
                (queue[-k-1] is None or ~(queue[-k-1]['gt_labels_3d']._data != -1).any()):
                return None
        return self.union2one(queue)

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def prepare_sliding_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """

        queue = []
        index_list = list(range(index-self.queue_length + 1, index))
        index_list = sorted(index_list)
        index_list.append(index)

        prev_scene_token = None
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict['scene_token'] != prev_scene_token:
                input_dict.update(dict(prev_exists=False))
                prev_scene_token = input_dict['scene_token']
            else:
                input_dict.update(dict(prev_exists=True))
            self.pre_pipeline(input_dict)


            example = self.pipeline(input_dict)
            queue.append(example)
        return self.union2one(queue)
    
    
    def union2one(self, queue):
        if self.queue_length == 1:
            for key in self.collect_keys:
                if key != 'img_metas':
                    queue[-1][key] = DC(torch.stack([each[key].data for each in queue]), cpu_only=False, stack=True, pad_dims=None)
            queue = queue[-1]
        else:
            for key in self.collect_keys:
                if key != 'img_metas':
                    queue[-1][key] = DC(torch.stack([each[key].data for each in queue]), cpu_only=False, stack=True, pad_dims=None)
                else:
                    queue[-1][key] = DC([each[key].data for each in queue], cpu_only=True)
            if not self.test_mode:
                for key in ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths']:
                    if key == 'gt_bboxes_3d':
                        queue[-1][key] = DC([each[key].data for each in queue], cpu_only=True)
                    else:
                        queue[-1][key] = DC([each[key].data for each in queue], cpu_only=False)
            queue = queue[-1]
        
        # print(queue['img']._data.size()) #torch.Size([8, 6, 3, 256, 512])
        # print(len(queue['img_metas']._data))
        # print(queue['prev_exists'])

        if self.root_path is not None and self.test_mode:
            ori_img = queue['img']._data
            filenames = queue['img_metas']._data[-1]['filename']
            # if self.test_mode:
            #     filenames = [filenames[i] for i in [0, 1, 5, 3, 4, 2]]
            mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
            std =  np.array([58.395, 57.12, 57.375], dtype=np.float32)
            pseudo_img_list = []
            for filename in filenames:
                img_path = self.root_path + filename.split("__")[1] + "_" + filename.split("/")[-1].replace(".jpg", "/")
                img_list = []
                for index in range(8):
                    name = img_path +"_00000"+ str(index)+".jpg"
                    # name = img_path +"_00000"+ str(index)+".png"
                    try:
                        img = mmcv.imread(name, "unchanged")
                        img = img.astype(np.float32)
                        img = mmcv.imnormalize(img, mean, std, True)
                        img_list.append(img)
                    except FileNotFoundError:
                        if not self.test_mode:
                            return None
                        else:
                            print(name)
                            img = np.zeros((ori_img.size(-2), ori_img.size(-1), 3))
                            img = img.astype(np.float32)
                            img = mmcv.imnormalize(img, mean, std, True)
                            img_list.append(img)
                        
                img = np.stack(img_list, axis=0)
                pseudo_img_list.append(img)
            pseudo_img = np.stack(pseudo_img_list, axis=1)
            pseudo_img = pseudo_img.astype(np.float32)

            pseudo_img = torch.from_numpy(pseudo_img).to(ori_img.device).permute(0, 1, 4, 2, 3)
            queue['img'] = DC(pseudo_img, cpu_only=False, stack=True, pad_dims=None)

        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch

        e2g_rotation = Quaternion(info['ego2global_rotation']).rotation_matrix
        e2g_translation = info['ego2global_translation']
        l2e_rotation = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        l2e_translation = info['lidar2ego_translation']
        e2g_matrix = convert_egopose_to_matrix_numpy(e2g_rotation, e2g_translation)
        l2e_matrix = convert_egopose_to_matrix_numpy(l2e_rotation, l2e_translation)
        ego_pose =  e2g_matrix @ l2e_matrix # lidar2global

        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego_pose=ego_pose,
            ego_pose_inv = ego_pose_inv,
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            img_timestamp = []
            for cam_type, cam_info in info['cams'].items():
                img_timestamp.append(cam_info['timestamp'] / 1e6)
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                cam2lidar_r = cam_info['sensor2lidar_rotation']
                cam2lidar_t = cam_info['sensor2lidar_translation']
                cam2lidar_rt = convert_egopose_to_matrix_numpy(cam2lidar_r, cam2lidar_t)
                lidar2cam_rt = invert_matrix_egopose_numpy(cam2lidar_rt)

                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt)
                intrinsics.append(viewpad)
                extrinsics.append(lidar2cam_rt)
                lidar2img_rts.append(lidar2img_rt)
                
            if not self.test_mode: # for seq_mode
                if self.seq_mode:
                    prev_exists  = not (index == 0 or self.flag[index - 1] != self.flag[index])
                else:
                    prev_exists  = not (index == 0)
            else:
                prev_exists = None

            input_dict.update(
                dict(
                    img_timestamp=img_timestamp,
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    prev_exists=prev_exists,
                ))
        if not self.test_mode:
            annos = self.get_ann_info(index)
            annos.update( 
                dict(
                    bboxes=info['bboxes2d'],
                    labels=info['labels2d'],
                    centers2d=info['centers2d'],
                    depths=info['depths'],
                    bboxes_ignore=info['bboxes_ignore'])
            )
            input_dict['ann_info'] = annos
            
        return input_dict


    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            if self.index_map is not None:
                index = self.index_map[idx]
            else:
                index = idx
            if not self.sliding_window:
                return self.prepare_test_data(index)
            else:
                return self.prepare_sliding_test_data(index)
        while True:
            if self.index_map is not None:
                index = self.index_map[idx]
            else:
                index = idx
                
            if self.root_path is not None:
                data = self.prepare_pseudotrain_data(index)
            else:
                data = self.prepare_train_data(index)

            if data is None:
                idx = self._rand_another(idx)
                continue

            return data
    
    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)
    
    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(det, self.with_velocity)
            if self.index_map is not None:
                sample_id = self.index_map[sample_id]
            sample_token = self.data_infos[sample_id]['token']
            if self.filter_tokens is not None and sample_token not in self.filter_tokens:
                continue

            boxes = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes,
                                             mapped_class_names,
                                             self.eval_detection_configs,
                                             self.eval_version)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        print(len(nusc_annos.keys()))
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix

def convert_egopose_to_matrix_numpy(rotation, translation):
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix