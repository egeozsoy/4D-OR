import bisect
import copy
import json
from collections import OrderedDict
from math import cos, sin
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


def coord_transform_OR_4D_to_human_pose_tool(arr: np.ndarray):
    orig_shape = arr.shape

    if len(orig_shape) == 1:
        arr = np.expand_dims(arr, 0)

    # flip Y and Z
    arr = arr[:, [0, 2, 1]]
    # Reverse new Y
    arr[:, 1] *= -1

    # change origin
    # arr[:, 0] += 1955
    # arr[:, 1] -= 100
    arr[:, 2] -= 1000

    # scale
    arr /= 25

    if len(orig_shape) == 1:
        return arr[0]
    return arr


def coord_transform_human_pose_tool_to_OR_4D(arr):
    # reverse of coord_transform_OR_4D_to_human_pose_tool
    arr *= 25

    arr[:, 2] += 1000

    arr[:, 1] *= -1

    arr = arr[:, [0, 2, 1]]

    return arr


def load_cam_infos(root_path: Path, cam_count=6):
    cam_infos = {}
    for c_idx in range(1, cam_count + 1):
        cam_json_path = root_path / f'camera0{c_idx}.json'
        with cam_json_path.open() as f:
            cam_info = json.load(f)['value0']
            intrinsics_json = cam_info['color_parameters']['intrinsics_matrix']
            intrinsics = np.asarray([[intrinsics_json['m00'], intrinsics_json['m10'], intrinsics_json['m20']],
                                     [intrinsics_json['m01'], intrinsics_json['m11'], intrinsics_json['m21']],
                                     [intrinsics_json['m02'], intrinsics_json['m12'], intrinsics_json['m22']]])

            extrinsics_json = cam_info['camera_pose']
            trans = extrinsics_json['translation']
            rot = extrinsics_json['rotation']
            extrinsics = np.zeros((4, 4), dtype=np.float32)
            rot_matrix = Rotation.from_quat([rot['x'], rot['y'], rot['z'], rot['w']]).as_matrix()
            extrinsics[:3, :3] = rot_matrix
            extrinsics[:, 3] = [trans['m00'], trans['m10'], trans['m20'], 1]

            color2depth_json = cam_info['color2depth_transform']
            trans = color2depth_json['translation']
            rot = color2depth_json['rotation']
            color2depth_transform = np.zeros((4, 4), dtype=np.float32)
            rot_matrix = Rotation.from_quat([rot['x'], rot['y'], rot['z'], rot['w']]).as_matrix()
            color2depth_transform[:3, :3] = rot_matrix
            color2depth_transform[:, 3] = [trans['m00'], trans['m10'], trans['m20'], 1]
            depth_extrinsics = np.copy(extrinsics)
            extrinsics = np.matmul(extrinsics, color2depth_transform)  # Extrinsics were given for the depth camera, convert them to color camera

            fov_x = cam_info['color_parameters']['fov_x']
            fov_y = cam_info['color_parameters']['fov_y']
            c_x = cam_info['color_parameters']['c_x']
            c_y = cam_info['color_parameters']['c_y']
            width = cam_info['color_parameters']['width']
            height = cam_info['color_parameters']['height']

            params = cam_info['color_parameters']['radial_distortion']
            radial_params = params['m00'], params['m10'], params['m20'], params['m30'], params['m40'], params['m50']
            params = cam_info['color_parameters']['tangential_distortion']
            tangential_params = params['m00'], params['m10']

            cam_infos[f'camera0{c_idx}'] = {'intrinsics': intrinsics, 'extrinsics': extrinsics, 'fov_x': fov_x, 'fov_y': fov_y,
                                            'c_x': c_x, 'c_y': c_y, 'width': width, 'height': height, 'radial_params': radial_params,
                                            'tangential_params': tangential_params, 'depth_extrinsics': depth_extrinsics}

    return cam_infos
