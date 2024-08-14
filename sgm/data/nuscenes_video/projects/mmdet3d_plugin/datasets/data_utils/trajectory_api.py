from random import sample
import numpy as np
import torch
from nuscenes.prediction import (PredictHelper,
                                 convert_local_coords_to_global,
                                 convert_global_coords_to_local)
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import to_tensor

class NuScenesTraj(object):
    def __init__(self,
                 nusc,
                 predict_steps,):
        super().__init__()
        self.nusc = nusc
        self.predict_steps = predict_steps
        self.predict_helper = PredictHelper(self.nusc)

    def get_traj_label(self, sample_token, ann_tokens):
        sample = self.nusc.get('sample', sample_token)
        fut_traj_all = []
        fut_traj_valid_mask_all = []
        sd_rec = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        pose_record = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])
        e2g_rotation = Quaternion(pose_record['rotation']).rotation_matrix
        e2g_translation = pose_record['translation']
        ego_pose = convert_egopose_to_matrix_numpy(e2g_rotation, e2g_translation)

        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
        for i, ann_token in enumerate(ann_tokens):
            instance_token = self.nusc.get('sample_annotation', ann_token)['instance_token']
            fut_traj_global= self.predict_helper.get_future_for_agent(instance_token, sample_token, seconds=6, in_agent_frame=False)

            fut_traj = np.zeros((self.predict_steps, 2))
            fut_traj_valid_mask = np.zeros((self.predict_steps, 2))
            if fut_traj_global.shape[0] > 0:
                fut_traj_scence_centric = fut_traj_global @ ego_pose_inv[:2, :2].T + ego_pose_inv[:2, 3]
                fut_traj[:fut_traj_scence_centric.shape[0], :] = fut_traj_scence_centric
                fut_traj_valid_mask[:fut_traj_scence_centric.shape[0], :] = 1


            fut_traj_all.append(fut_traj)		
            fut_traj_valid_mask_all.append(fut_traj_valid_mask)		
        if len(ann_tokens) > 0:		
            fut_traj_all = np.stack(fut_traj_all, axis=0)		
            fut_traj_valid_mask_all = np.stack(fut_traj_valid_mask_all, axis=0)		
        else:		
            fut_traj_all = np.zeros((0, self.predict_steps, 2))		
            fut_traj_valid_mask_all = np.zeros((0, self.predict_steps, 2))		
        return fut_traj_all, fut_traj_valid_mask_all

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