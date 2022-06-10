import open3d as o3d
import numpy as np
from copy import deepcopy
from math import sin, cos
from helpers.configurations import DEPTH_SCALING, OR_4D_DATA_ROOT_PATH


def vec_ang(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)


def get_object_poses(registered_objects):
    poses = []
    names = []
    for object_scan_path, transformation in registered_objects.items():
        transformation[:3, 3] = transformation[:3, 3] * DEPTH_SCALING
        tmp = o3d.io.read_point_cloud(str(object_scan_path))
        object_name = object_scan_path.split("/")[3]
        object_scan = o3d.geometry.PointCloud()
        object_scan.points = o3d.utility.Vector3dVector(np.asarray(tmp.points))
        object_scan.colors = tmp.colors
        object_scan_temp = deepcopy(object_scan)
        object_scan_temp.transform(transformation)

        poses.append(object_scan_temp)
        names.append(object_name)

    return poses, names

# def get_transform_matrix(rot_deg=0., rot_axis='x'):
#     rot_deg = np.deg2rad(rot_deg)
#     transform_matrix = np.eye(4, dtype=np.float64)
#     if rot_axis == 'x':
#         transform_matrix[1:3, 1:3] = np.array([[cos(rot_deg), -sin(rot_deg)],
#                                                [sin(rot_deg), cos(rot_deg)]], dtype=np.float64)
#
#     if rot_axis == 'y':
#         transform_matrix[[0, 2]] = np.array([[cos(rot_deg), 0, sin(rot_deg), 0],
#                                              [-sin(rot_deg), 0, cos(rot_deg), 0]], dtype=np.float64)
#     if rot_axis == 'z':
#         transform_matrix[:2, :2] = np.array([[cos(rot_deg), -sin(rot_deg)],
#                                              [sin(rot_deg), cos(rot_deg)]], dtype=np.float64)
#
#     return transform_matrix
