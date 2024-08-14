import cv2
import numpy as np
from torch.utils.data import Dataset
import random
from einops import rearrange
import numpy as np
import sys
sys.path.append('sgm/data/nuscenes_video')
print(sys.path)
from projects.mmdet3d_plugin.datasets import CustomNuScenesDataset
import random
from render import Renderer
from mmcv.parallel import DataContainer
from shapely.geometry import MultiPoint, box
from typing import List, Tuple, Union
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
import copy
import cv2
from mmdet.datasets.builder import PIPELINES

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (3840, 2160)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None

def bbox3d_to_8corners(bbox3des):
    # 3d bbox to 8 corners
    box_corners = []
    for bbox3d in bbox3des:
        xyz_wlh_rot = bbox3d[[0, 1, 2, 4, 3, 5, 6]]  # exchange w,l
        rot = Quaternion(axis=[0, 0, 1], radians=xyz_wlh_rot[6])  # rot_z
        rot_box = Box(center=xyz_wlh_rot[0:3], size=xyz_wlh_rot[3:6], orientation=rot)
        box_corner = rot_box.corners().T
        box_corners.append(box_corner)
        # 8 point order： https://transformer.iap.wh-a.brainpp.cn/t/topic/137/3
    box_corners = np.array(box_corners)
    return box_corners

def draw_rect(img, selected_corners, color, linewidth):
    prev = selected_corners[-1]
    for corner in selected_corners:
        cv2.line(img,
                    (int(prev[0]), int(prev[1])),
                    (int(corner[0]), int(corner[1])),
                    color, linewidth)
        prev = corner

# data_root = "/data/Dataset/nuimages/"
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

def list_move_right(A,a):
    for i in range(a):
        A.insert(0,A.pop())
    return A

prompt_list = [
    "This portrays an urban road panorama, composed from a jointure of six encompassing viewpoint images.  The whole scene contains {} targets, comprising ",
    "The encompassed illustration discloses an urban roadway scenery pieced together from six wide-angle snapshots.  The entire depiction embraces {} objectives, inclusive of ",
    "We observe an urban street spectacle, arranged from the composition of six all-encompassing perspective graphics. The collective stage comprises {} features, including ",
    "We're presented with an urban route visualization, amalgamated from six encompassing angle images. The entire layout thus, contains {} targets, comprising ",
    "This reveals a city thoroughfare tableau, composed of a fusion of six panoramic imagery.  This comprehensive depiction incorporates {} goals, including ",
    "It embodies an urban lane portrait, incontrarintegrated from six encompassing angle snips. The whole scene accommodates {} subject matters, including ",
    "This unfolds an urban road sight, consolidated from partials of six encompassing viewpoint snapshots. The integrated framework invites {} subjects, incorporating ",
    "Within, we find an urban theme combining components from six sweeping viewpoint captures. The all-inclusive image contains {} targets including ",
    "This rendering is an urban avenue spectacle, masterfully stiched from six encompassing viewpoint visuals. The inclusive display holds {} subjects, embracing ",
    "Embodied in an urban route tableau, synthesized from six surround-view visuals. The unfolding tableau harbors {} subjects, comprising ",
    "This captures a metropolitan road scenario, meticulously constructed from a montage of six surrounding viewpoint images. The entirety of the scene contains {} targets, inclusive of ",
    "Here is an illustration of a cosmopolitan street view, artfully crafted from six peripheral vantage point pictures. The full layout accommodates {} objectives, including ",
    "Presented is a city locale framework, creatively assembled using six encompassing angular perspectives. The gathered tableau incorporates {} targets, encompassing ",
    "What you see is a city street setting, ingeniously put together from six panoramic snapshots. This amalgamated scene houses {} objectives, inclusive of ",
    "We have an urban street decor, deftly stitched together utilizing six peripheral perspective images. The comprehensive scene involves {} objects, embodying ",
    "This presents an urban road environment, constructed with the fusion of six surrounding perspective images. The whole scene contains {} targets, including ",
    "Here we have a city road tableau, synthesized from an amalgamation of six surrounding view images. Overall, the scene incorporates {} objects of interest, including ",
    "We're presented with an urban thoroughfare scene, born of the merging of six circumambient visual representations. The panorama contains {} landmarks, including ",
    "This is a representation of a city street setting, composing of six pictures portraying different angles. The complete landscape includes {} principal targets, including ",
    "Displayed is a metropolitan roadway scenario, composed of six peripheral perspective images combined. The holistic scenario encloses {} prime objectives, embodying ",
]

point_cloud_range = [-30, -30, -5.0, 30, 30, 3.0]
voxel_size = [0.2, 0.2, 8]
coords_dim = 2 # polylines coordinates dimension, 2 or 3
# bev configs
roi_size = (60, 30) # bev range, 60m in x-axis, 30m in y-axis
canvas_size = (200, 100) # bev feature size
cat2id_map = {
    'ped_crossing': 0,
    'divider': 1,
    'boundary': 2,
}
num_class_map = max(list(cat2id_map.values())) + 1
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
img_norm_cfg = dict(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        to_rgb=True)
collect_keys=['lidar2img', 'intrinsics', 'extrinsics','timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv']
collect_labels = None
num_frame_losses = 1
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,)

class MyDataset(Dataset):
    def __init__(self,
        image_size=(640, 320),
        point_cloud_range = [-35, -35, -5.0, 35, 35, 3.0],
        ida_aug_conf = {
            "resize_lim": (0.32, 0.32),
            "final_dim": (256, 512),
            "bot_pct_lim": (0.0, 0.0),
            "rot_lim": (0.0, 0.0),
            "H": 900,
            "W": 1600,
            "rand_flip": False,
        },
        split="train",
        queue_length=1,
        shift_view=False,
        random_shift=False,
        render_pose=False,
        bda_aug=False,
        source_img=False,
        sample_num=20,
        repeat_cond_frames=False,
        use_last_frame=True,
        ):
        self.repeat_cond_frames=repeat_cond_frames
        self.use_last_frame=use_last_frame
        if repeat_cond_frames:
            print("!warining, repeat cond frames")

        pipeline = [
        dict(
            type='VectorizeMap',
            coords_dim=coords_dim,
            roi_size=roi_size,
            sample_num=sample_num,
            normalize=True,
        ),
        dict(
            type='PolygonizeLocalMapBbox',
            canvas_size=canvas_size,
            coord_dim=2,
            num_class=num_class_map,
            threshold=4/200,
        ),
        dict(type='LoadMultiViewImageFromFiles', to_float32=True),
        dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
            with_label=True, with_bbox_depth=True),
        dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='CustomObjectNameFilter', classes=class_names),
        dict(type='ResizeCropFlipRotImage', data_aug_conf = ida_aug_conf, training=True),
        dict(type='NormalizeMultiviewImage', **img_norm_cfg),
        dict(type='PadMultiViewImage', size_divisor=32),
        dict(type='PETRFormatBundle3D', class_names=class_names, collect_keys=collect_keys + ['prev_exists']),
        dict(type='Collect3D', keys=['sample_idx', 'gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'prev_exists',  'vectors'] + collect_keys,
            meta_keys=('token', 'filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d','gt_labels_3d'))
    ]

        ann_file = "data/nuscenes/nuscenes2d_ego_temporal_infos_{}.pkl".format(split)
        self.ann_file=ann_file
        print('ann_file is ', ann_file)
        self.data = CustomNuScenesDataset(
            roi_size=roi_size,
            cat2id_map=cat2id_map,
            data_root="data/nuscenes/",
            ann_file=ann_file,
            num_frame_losses=queue_length,
            pipeline=pipeline,
            classes=class_names,
            modality=input_modality,
            collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas', 'vectors'],
            collect_labels=collect_labels,
            queue_length=queue_length,
            test_mode=False,
            overlap_test=False,
            eval_mod=['map'],
            use_valid_flag=True,
            filter_empty_gt=False,
            box_type_3d='LiDAR')

        self.queue_length = queue_length
        self.image_size = (ida_aug_conf["final_dim"][1], ida_aug_conf["final_dim"][0])
        self.classes = self.data.CLASSES
        self.split = split
        self.shift_view = shift_view
        self.render_pose = render_pose
        self.random_shift = random_shift
        self.source_img = source_img
        self.colors = np.array([
                [255, 255, 255],
                [128, 64, 128],
                [244, 35, 232],
                [70, 70, 70],
                [102, 102, 156],
                [190, 153, 153],
                [153, 153, 153],
                [250, 170, 30],
                [220, 220, 0],
                [107, 142, 35],
                [152, 251, 152],
                [0, 130, 180],
                [220, 20, 60],
                [255, 0, 0],
                [0, 0, 142],
                [0, 0, 70],
                [0, 60, 100],
                [0, 80, 100],
                [0, 0, 230],
                [119, 11, 32]])
        self.render = Renderer(cat2id_map, roi_size, 'nusc')

        self.view_colors = {
            'CAM_FRONT':[0, 130, 180],
            'CAM_FRONT_RIGHT':[220, 20, 60],
            'CAM_BACK_RIGHT':[255, 0, 0],
            'CAM_BACK':[0, 0, 142],
            'CAM_BACK_LEFT':[0, 60, 100],
            'CAM_FRONT_LEFT': [119, 11, 32]
        }
        
        self.viewid = {
                'CAM_FRONT':0,
                'CAM_FRONT_RIGHT':1,
                'CAM_BACK_RIGHT':5,
                'CAM_BACK':3,
                'CAM_BACK_LEFT':4,
                'CAM_FRONT_LEFT':2
            }

    def __len__(self):
        return len(self.data)

    def generate_prompts(self, label, bbox):
        label = [self.classes[id] for id in label]
        label_name = label

        prompt = random.sample(prompt_list, 1)[0]
        prompt_label = ', '.join(label_name)
        prompt = prompt.format(str(len(label))) + prompt_label 
        
        return prompt, prompt_label

    def draw_bboxes(self, target, bboxes, labels, depths, colors, thickness=12):
        
        img = np.ones((target.shape[0], target.shape[1], len(self.classes))) * 255 
        img = img.copy().astype(np.uint8)
        if labels is None or len(labels) == 0:
            return img

        for i, name in enumerate(self.classes):
            mask = (labels == i)
            lab = labels[mask]
            dep = depths[mask]
            if bboxes is not None: bbox = bboxes[mask] 
            if bboxes is None or len(bbox) == 0:
                continue
            dep = dep * 3
            for j in range(len(bbox)):
                xmin,ymin,xmax,ymax = bbox[j]
                img[int(ymin) : int(ymax), int(xmin) : int(xmax), i] = np.where(img[int(ymin) : int(ymax), int(xmin) : int(xmax), i] > dep[j], dep[j], img[int(ymin) : int(ymax), int(xmin) : int(xmax), i])

        return img 
    
    def draw_corners(self, target, corners, labels, depths2d, colors, linewidth=4):
        
        img = np.ones((target.shape[0], target.shape[1], 3)) * 255 
        img = img.copy().astype(np.uint8)

        if corners is None or len(corners) == 0:
            return img

        sort_indexes = np.argsort(depths2d)[::-1]
        corners = corners[sort_indexes]
        labels = labels[sort_indexes]
        depths2d = depths2d[sort_indexes]
        for j in range(len(corners)):
            color = colors[labels[j] + 1]
            color = (int(color[0]), int(color[1]), int(color[2]))

            points = corners[j, [0, 1, 2, 3]]
            # points =  np.array([[int(corner[j, 0, 0]), int(corner[j, 0, 1])], [int(corner[j, 1, 0]), int(corner[j, 1, 1])], [int(corner[j, 2, 0]), int(corner[j, 2, 1])], [int(corner[j, 3, 0]), int(corner[j, 3, 1])]])
            points =  np.array([[int(corners[j, 4, 0]), int(corners[j, 4, 1])], [int(corners[j, 5, 0]), int(corners[j, 5, 1])], [int(corners[j, 6, 0]), int(corners[j, 6, 1])], [int(corners[j, 7, 0]), int(corners[j, 7, 1])]])
            points = points.reshape(-1, 1, 2)
            points[..., 0] = np.clip(points[..., 0], 0, target.shape[1]) 
            points[..., 1] = np.clip(points[..., 1], 0, target.shape[0]) 
            ori_color = (int(color[0]*0.5 + 255*0.5), int(color[1]*0.5 + 255*0.5), int(color[2]*0.5 + 255*0.5))
            cv2.fillPoly(img, [points], ori_color)

            for i in range(4):
                cv2.line(img,
                    (int(corners[j][i][0]), int(corners[j][i][1])),
                    (int(corners[j][i + 4][0]), int(corners[j][i + 4][1])),
                    color[::-1], linewidth)

            draw_rect(img, corners[j][:4], color[::-1], linewidth)
            draw_rect(img, corners[j][4:], color[::-1], linewidth)
        
        return img 

    def render_views(self, shapes, camera_views):
        img_list = []
        for i, view in enumerate(camera_views):
            img = np.zeros((shapes[0], shapes[1], 3))
            img = img.copy().astype(np.uint8)
            color = np.array(self.view_colors[view])
            img = img + color[None, None, :]
            img_list.append(img)
        return img_list
    
    def render_map(self, renderer, data, imgs, lidar2imgs, camera_views,vectors_frame, out_dir='visualize/'):
        '''Visualize ground-truth.

        Args:
            idx (int): index of sample.
            out_dir (str): output directory.
        '''
        roi_size = (60, 30)
        lidar2img = []
        for i, view in enumerate(camera_views):
            lidar2img.append(lidar2imgs[self.viewid[view]])
        lidar2img = np.asarray(lidar2img) # (N, 4, 4)

        if 'vectors' in data:
            vectors = vectors_frame
            if isinstance(vectors, DataContainer):
                vectors = vectors.data
            roi_size = np.array(roi_size)
            origin = -np.array([roi_size[0]/2, roi_size[1]/2])

            for k, vector_list in vectors.items():
                for i, v in enumerate(vector_list):
                    v[:, :2] = v[:, :2] * (roi_size + 2) + origin
                    vector_list[i] = v
            img_list = renderer.render_camera_views_from_vectors(vectors, imgs, 
                lidar2img, 4, None)
        return img_list, vectors


    def render_directions(self, shapes, img2lidars, camera_views):

        eps = 1e-5
        N = len(img2lidars)
        H, W, _ = shapes
        coords_h = np.arange(H)
        coords_w = np.arange(W)
        # coords_d = np.array([1.0])
        coords_d = np.array([1.0, 2.0])

        D = coords_d.shape[0]
        coords = np.stack(np.meshgrid(coords_w, coords_h, coords_d)).transpose((1, 2, 3, 0)) # W, H, D, 3
        coords = np.concatenate((coords, np.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * np.maximum(coords[..., 2:3], np.ones_like(coords[..., 2:3])*eps)
        img2lidar = []
        for i, view in enumerate(camera_views):
            img2lidar.append(img2lidars[self.viewid[view]])
        img2lidar = np.asarray(img2lidar) # (N, 4, 4)
        coords = coords.reshape(1, W, H, D, 4, 1)
        img2lidar = img2lidar.reshape(N, 1, 1, 1, 4, 4)

        coords3d = np.matmul(img2lidar, coords).squeeze(-1)[..., :3]
        coords3d = coords3d.transpose((0, 2, 1, 3, 4))

        directions = coords3d[:, :, :, 1, :] - coords3d[:, :, :, 0, :]
        coords3d = (directions - directions.min()) / (directions.max() - directions.min()) * 255
        coords3d = coords3d.copy().astype(np.uint8)
        coords3d = [cord3d for cord3d in coords3d]


        return coords3d
    
    def _get_2d_annos(self, shape, source_bbox_3d, source_corner_3d, source_label_3d, lidar2img):
        gt_bboxes_3d = source_bbox_3d
        gt_label_3d = source_label_3d
        corners_3d = source_corner_3d
        num_bbox = corners_3d.shape[0]
        pts_4d = np.concatenate([corners_3d.reshape(-1, 3), np.ones((num_bbox * 8, 1))], axis=-1)

        gt_bbox2d = []
        gt_depth2d = []
        gt_label2d = []
        gt_corners3d = []
        for i in range(6):
            lidar2img_rt = np.array(lidar2img[i])
            pts_2d = pts_4d @ lidar2img_rt.T
            pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=0.1, a_max=51.2)
            pts_2d[:, 0] /= pts_2d[:, 2]
            pts_2d[:, 1] /= pts_2d[:, 2]
            
            H, W = shape[0], shape[1]
            imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)
            imgfov_pts_depth = pts_2d[..., 2].reshape(num_bbox, 8)
            mask = imgfov_pts_depth.mean(1) > 0.1

            if mask.sum() == 0:
                gt_bbox2d.append([])
                gt_depth2d.append([])
                gt_label2d.append([]) 
                gt_corners3d.append([])
                continue

            imgfov_pts_2d = imgfov_pts_2d[mask]
            imgfov_pts_depth = imgfov_pts_depth[mask]
            imgfov_pts_label= gt_label_3d[mask]

            bbox = []
            label = []
            depth = []
            corners3d = []
            for j, corner_coord in enumerate(imgfov_pts_2d):
                final_coords = post_process_coords(corner_coord, imsize = (W,H))
                if final_coords is None:
                    continue
                else:
                    min_x, min_y, max_x, max_y = final_coords
                    if ((max_x - min_x) >W-100) and ((max_y - min_y)>H-100):
                        continue
                    bbox.append([min_x, min_y, max_x, max_y])
                    label.append(imgfov_pts_label[j])
                    depth.append(imgfov_pts_depth[j].mean())
                    corners3d.append(copy.deepcopy(corner_coord))
            gt_bbox2d.append(np.array(bbox))
            gt_depth2d.append(np.array(depth))
            gt_label2d.append(np.array(label)) 
            gt_corners3d.append(np.array(corners3d)) 
        bbox2d_info = {
            'gt_bbox2d' : gt_bbox2d,
            'gt_depth2d' : gt_depth2d,
            'gt_label2d' : gt_label2d,
            'gt_corners3d': gt_corners3d
        }

        return bbox2d_info

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = range(len(self))
        return np.random.choice(pool)

    def __getitem__(self, idx):
        while True:
            item = self.data[idx]
            if item['prev_exists']._data.sum() != (len(item['prev_exists']._data) - 1):
                idx = self._rand_another(idx)
                continue
            break
        target_frames=[]
        source_frames=[]
        for frame_id in range(self.queue_length):
            source_img = item['img']._data[frame_id].numpy() #torch.Size([6, 3, 256, 704]) 
            source_label_3d = item['gt_labels_3d']._data[frame_id].numpy() # torch.Size([32])  
            source_bbox_3d = item['gt_bboxes_3d']._data[frame_id].tensor.numpy() # torch.Size([32, 9])
            img2lidar  = item['lidar2img']._data[frame_id].inverse().numpy() # torch.Size([6, 4, 4])
            lidar2img  = item['lidar2img']._data[frame_id].numpy() # torch.Size([6, 4, 4])
            source_corner_3d = item['gt_bboxes_3d']._data[frame_id].corners.numpy()
            vectors_frame = item['vectors']._data[frame_id]
            bbox2d_info = self._get_2d_annos((source_img.shape[2], source_img.shape[3]), source_bbox_3d, source_corner_3d, source_label_3d, lidar2img)
            source_label_2d = bbox2d_info['gt_label2d']
            source_bbox_2d = bbox2d_info['gt_bbox2d']
            source_depth_2d = bbox2d_info['gt_depth2d']
            source_corner_2d = bbox2d_info['gt_corners3d']

            camera_views = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
            if self.shift_view and self.split == "train":
                if self.random_shift:
                    random.shuffle(camera_views)
                else:
                    camera_views = list_move_right(camera_views, random.choice(range(len(camera_views)))) 
            img_list = []
            source_list = []
            for view in camera_views:
                img = source_img[self.viewid[view]].transpose((1,2,0))
                
                bboxes2d = source_bbox_2d[self.viewid[view]]
                labels2d = source_label_2d[self.viewid[view]]
                depths2d = source_depth_2d[self.viewid[view]]
                corners2d = source_corner_2d[self.viewid[view]]
                source = self.draw_bboxes(img, bboxes2d, labels2d, depths2d, self.colors)
                source_corner = self.draw_corners(img, corners2d, labels2d, depths2d, self.colors, linewidth=2) #for 512
                source = np.concatenate([source_corner, source], -1)
                img_list.append(img)
                source_list.append(source)
            
            map_list, vectors = self.render_map(self.render, item, img_list, lidar2img, camera_views, vectors_frame)

            if self.render_pose:
                render_list = self.render_directions(img.shape, img2lidar, camera_views)
            else:
                render_list = self.render_views(img.shape, camera_views)
            
            target = np.concatenate(img_list, axis=1)
            source = np.concatenate(source_list, axis=1)
            render_pe = np.concatenate(render_list, axis=1)
            render_map = np.concatenate(map_list, axis=1)
            prompt, prompt_label = self.generate_prompts(source_label_3d, source_bbox_3d)
            source = np.concatenate([source, render_map, render_pe], -1) 
            target_frames.append(target)
            source_frames.append(source)
        target = np.stack(target_frames, axis=0) 
        target = rearrange(target, 't h w c -> t c h w')
        source = np.stack(source_frames, axis=0) 
        source = rearrange(source, 't h w c -> t c h w')
      
        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
        target = (target.astype(np.float32) / 127.5) - 1.0

        filenames=[]
        for frame_id in range(self.queue_length):
            filename=item['img_metas']._data[frame_id]['filename']
            filename[2],filename[5] = filename[5],filename[2]
            filenames.append(filename)
        if not self.repeat_cond_frames:
            imgs_cond = np.zeros_like(target)
            if self.use_last_frame:
                cond_img = target[-1, ...]
                imgs_cond[-1, ...] = cond_img
            else: #use first frame
                cond_img = target[0, ...]
                imgs_cond[0, ...] = cond_img
        else:
            if self.use_last_frame:
                cond_img = target[-1, ...]
            else:
                cond_img = target[0, ...]
            imgs_cond = np.tile(cond_img, (8, 1, 1, 1))
        return dict(jpg=target, txt=prompt, cond_img=source, final_cond_zero=imgs_cond, filenames=filenames)


if __name__=='__main__':
    print("build dataset")
    data = MyDataset(split="val", image_size=(512, 320), queue_length=8,render_pose=True,bda_aug=False)
    dataidx=data.__getitem__(47)
    print("finished")

