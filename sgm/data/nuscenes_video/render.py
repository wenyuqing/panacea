import os.path as osp
import os
import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from shapely.geometry import LineString
from mmcv.parallel import DataContainer
from copy import deepcopy
import torch

def remove_nan_values(uv):
    is_u_valid = np.logical_not(np.isnan(uv[:, 0]))
    is_v_valid = np.logical_not(np.isnan(uv[:, 1]))
    is_uv_valid = np.logical_and(is_u_valid, is_v_valid)

    uv_valid = uv[is_uv_valid]
    return uv_valid

def points_ego2img(pts_ego, lidar2img):
    pts_ego_4d = np.concatenate([pts_ego, np.ones([len(pts_ego), 1])], axis=-1)
    pts_cam_4d = lidar2img @ pts_ego_4d.T  
    uv = pts_cam_4d[:3, :].T
    uv = remove_nan_values(uv)
    depth = uv[:, 2]
    uv = uv[:, :2] / uv[:, 2].reshape(-1, 1)

    return uv, depth

def interp_fixed_num(line, num_pts):
    ''' Interpolate a line to fixed number of points.
    
    Args:
        line (LineString): line
    
    Returns:
        points (array): interpolated points, shape (N, 2)
    '''

    distances = np.linspace(0, line.length, num_pts)
    sampled_points = np.array([list(line.interpolate(distance).coords) 
        for distance in distances]).squeeze()

    return sampled_points

def draw_polyline_ego_on_img(polyline_ego, img_bgr, lidar2img, color_bgr, thickness):

    if polyline_ego.shape[1] == 2:
        zeros = np.zeros((polyline_ego.shape[0], 1))
        polyline_ego = np.concatenate([polyline_ego, zeros], axis=1)
    polyline_ego = interp_fixed_num(line=LineString(polyline_ego), num_pts=200)
    uv, depth = points_ego2img(polyline_ego, lidar2img)

    h, w, c = img_bgr.shape

    is_valid_x = np.logical_and(0 <= uv[:, 0], uv[:, 0] < w - 1)
    is_valid_y = np.logical_and(0 <= uv[:, 1], uv[:, 1] < h - 1)
    is_valid_z = depth > 0
    is_valid_points = np.logical_and.reduce([is_valid_x, is_valid_y, is_valid_z])

    if is_valid_points.sum() == 0:
        return img_bgr
    uv = np.round(uv[is_valid_points]).astype(np.int32)
    img = draw_visible_polyline_cv2(
        copy.deepcopy(uv),
        valid_pts_bool=np.ones((len(uv), 1), dtype=bool),
        image=img_bgr,
        color=color_bgr,
        thickness_px=thickness,
    )

    return img

def draw_visible_polyline_cv2(line, valid_pts_bool, image, color, thickness_px):
    """Draw a polyline onto an image using given line segments.

    Args:
        line: Array of shape (K, 2) representing the coordinates of line.
        valid_pts_bool: Array of shape (K,) representing which polyline coordinates are valid for rendering.
            For example, if the coordinate is occluded, a user might specify that it is invalid.
            Line segments touching an invalid vertex will not be rendered.
        image: Array of shape (H, W, 3), representing a 3-channel BGR image
        color: Tuple of shape (3,) with a BGR format color
        thickness_px: thickness (in pixels) to use when rendering the polyline.
    """
    line = np.round(line).astype(int)  # type: ignore
    for i in range(len(line) - 1):

        if (not valid_pts_bool[i]) or (not valid_pts_bool[i + 1]):
            continue

        x1 = line[i][0]
        y1 = line[i][1]
        x2 = line[i + 1][0]
        y2 = line[i + 1][1]

        # Use anti-aliasing (AA) for curves
        image = cv2.line(image, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=thickness_px, lineType=cv2.LINE_AA)

    return image 

COLOR_MAPS_BGR = {
    # bgr colors
    'divider': (0, 0, 255),
    'boundary': (0, 255, 0),
    'ped_crossing': (255, 0, 0),
    'centerline': (51, 183, 255),
    'drivable_area': (171, 255, 255)
}

COLOR_MAPS_PLT = {
    'divider': 'r',
    'boundary': 'g',
    'ped_crossing': 'b',
    'centerline': 'orange',
    'drivable_area': 'y',
}

CAM_NAMES_NUSC = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',]

class Renderer(object):
    """Render map elements on image views.

    Args:
        cat2id (dict): category to class id
        roi_size (tuple): bev range
        dataset (str): 'av2' or 'nusc'
    """

    def __init__(self, cat2id, roi_size, dataset='nusc'):
        self.roi_size = roi_size
        self.cat2id = cat2id
        self.id2cat = {v: k for k, v in cat2id.items()}
        self.cam_names = CAM_NAMES_NUSC

    def render_bev_from_vectors(self, vectors, out_dir):
        '''Render bev segmentation using vectorized map elements.
        
        Args:
            vectors (dict): dict of vectorized map elements.
            out_dir (str): output directory
        '''

        car_img = Image.open('./visualize/car.png')
        map_path = os.path.join(out_dir, 'map.jpg')

        plt.figure(figsize=(self.roi_size[0], self.roi_size[1]))
        plt.xlim(-self.roi_size[0] / 2, self.roi_size[0] / 2)
        plt.ylim(-self.roi_size[1] / 2, self.roi_size[1] / 2)
        plt.axis('off')
        plt.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2])

        for label, vector_list in vectors.items():
            cat = self.id2cat[label]
            color = COLOR_MAPS_PLT[cat]
            for vector in vector_list:
                pts = vector[:, :2]
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], angles='xy', color=color,
                    scale_units='xy', scale=1)

        plt.savefig(map_path, bbox_inches='tight', dpi=40)
        plt.close()
        
    def render_camera_views_from_vectors(self, vectors, imgs, lidar2imgs, 
            thickness, out_dir=None):
        '''Project vectorized map elements to camera views.
        
        Args:
            vectors (dict): dict of vectorized map elements.
            imgs (tensor): images in bgr color.
            extrinsics (array): ego2img extrinsics, shape (4, 4)
            intrinsics (array): intrinsics, shape (3, 3) 
            thickness (int): thickness of lines to draw on images.
            out_dir (str): output directory
        '''
        img_list = []
        for i in range(len(imgs)):
            # img = imgs[i]
            img = np.ones((imgs[i].shape[0], imgs[i].shape[1], 3)) * 255 
            lidar2img = lidar2imgs[i]
            img_bgr = copy.deepcopy(img)

            for label, vector_list in vectors.items():
                cat = self.id2cat[label]
                color = COLOR_MAPS_BGR[cat]
                for vector in vector_list:
                    img_bgr = np.ascontiguousarray(img_bgr)
                    img = draw_polyline_ego_on_img(vector, img_bgr, lidar2img, 
                       color, thickness)
            img_list.append(img)
            if out_dir is not None:
                out_path = osp.join(out_dir, self.cam_names[i]) + '.jpg'
                cv2.imwrite(out_path, img_bgr)
        
        return img_list

    def render_bev_from_mask(self, semantic_mask, out_dir):
        '''Render bev segmentation from semantic_mask.
        
        Args:
            semantic_mask (array): semantic mask.
            out_dir (str): output directory
        '''

        c, h, w = semantic_mask.shape
        bev_img = np.ones((3, h, w), dtype=np.uint8) * 255
        if 'drivable_area' in self.cat2id:
            drivable_area_mask = semantic_mask[self.cat2id['drivable_area']]
            bev_img[:, drivable_area_mask == 1] = \
                    np.array(COLOR_MAPS_BGR['drivable_area']).reshape(3, 1)

        for label in range(c):
            cat = self.id2cat[label]
            if cat == 'drivable_area':
                continue
            mask = semantic_mask[label]
            valid = mask == 1
            bev_img[:, valid] = np.array(COLOR_MAPS_BGR[cat]).reshape(3, 1)
        
        bev_img_flipud = np.array([np.flipud(i) for i in bev_img], dtype=np.uint8)
        out_path = osp.join(out_dir, 'semantic_map.jpg')
        cv2.imwrite(out_path, bev_img_flipud.transpose((1, 2, 0)))


def show_gt(renderer, data, cfg=None, out_dir='visualize/'):
    '''Visualize ground-truth.

    Args:
        idx (int): index of sample.
        out_dir (str): output directory.
    '''
    imgs = data["img"]._data
    imgs = imgs.flatten(0, 1)
    mean = torch.from_numpy(np.array(cfg.img_norm_cfg['mean'])).view(1, -1, 1, 1).to(imgs.device)
    std = torch.from_numpy(np.array(cfg.img_norm_cfg['std'])).view(1, -1, 1, 1).to(imgs.device)
    imgs = imgs.clone()*std + mean
    imgs = [img.permute(1, 2, 0).cpu().detach().numpy()[:,:,::-1] for img in imgs]
    cam_extrinsics = data['extrinsics']._data[0].cpu().numpy()
    cam_intrinsics = data['intrinsics']._data[0].cpu().numpy()[:, :3, :3]

    if 'vectors' in data:
        vectors = data['vectors']
        if isinstance(vectors, DataContainer):
            vectors = vectors.data
        roi_size = np.array(cfg.roi_size)
        origin = -np.array([roi_size[0]/2, roi_size[1]/2])

        for k, vector_list in vectors.items():
            for i, v in enumerate(vector_list):
                v[:, :2] = v[:, :2] * (roi_size + 2) + origin
                vector_list[i] = v
        
        renderer.render_bev_from_vectors(vectors, out_dir)
        renderer.render_camera_views_from_vectors(vectors, imgs, 
            cam_extrinsics, cam_intrinsics, 2, out_dir)

    if 'semantic_mask' in data:
        semantic_mask = data['semantic_mask']
        if isinstance(semantic_mask, DataContainer):
            semantic_mask = semantic_mask.data
        
        renderer.render_bev_from_mask(semantic_mask, out_dir)


def show_result(renderer, dataset, submission, idx, score_thr=0, cfg=None, out_dir='visualize/'):
    '''Visualize prediction result.

    Args:
        idx (int): index of sample.
        submission (dict): prediction results.
        score_thr (float): threshold to filter prediction results.
        out_dir (str): output directory.
    '''

    meta = submission['meta']
    output_format = meta['output_format']
    data = dataset.__getitem__(idx)

    if isinstance(data['img_metas'], list):
        token = data['img_metas'][0]._data['token']
    else:
        token = data['img_metas']._data[0]['token']
    results = submission['results'][token]

    imgs = data["img"]._data
    imgs = imgs.flatten(0, 1)
    mean = torch.from_numpy(np.array(cfg.img_norm_cfg['mean'])).view(1, -1, 1, 1).to(imgs.device)
    std = torch.from_numpy(np.array(cfg.img_norm_cfg['std'])).view(1, -1, 1, 1).to(imgs.device)
    imgs = imgs.clone()*std + mean
    imgs = [img.permute(1, 2, 0).cpu().detach().numpy()[:,:,::-1] for img in imgs]
    cam_extrinsics = data['extrinsics']._data[0].cpu().numpy()
    cam_intrinsics = data['intrinsics']._data[0].cpu().numpy()[:, :3, :3]

    if output_format == 'raster':
        semantic_mask = results['semantic_mask'].numpy()
        renderer.render_bev_from_mask(semantic_mask, out_dir)
    
    elif output_format == 'vector':
        vectors = {label: [] for label in cfg.cat2id_map.values()}
        for i in range(len(results['labels'])):
            score = results['scores'][i]
            label = results['labels'][i]
            v = results['vectors'][i]
            if score > score_thr:
                vectors[label].append(v)
            
        renderer.render_bev_from_vectors(vectors, out_dir)
        renderer.render_camera_views_from_vectors(vectors, imgs, 
                cam_extrinsics, cam_intrinsics, 2, out_dir)

