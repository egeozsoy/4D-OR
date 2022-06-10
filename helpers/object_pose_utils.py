import copy

import numpy as np
import open3d as o3d

from helpers.configurations import DEPTH_SCALING, OBJECT_COLOR_MAP


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
        object_scan_temp = copy.deepcopy(object_scan)
        object_scan_temp.transform(transformation)
        object_scan_temp.paint_uniform_color(OBJECT_COLOR_MAP[object_name])

        poses.append(object_scan_temp)
        names.append(object_name)

    return poses, names
