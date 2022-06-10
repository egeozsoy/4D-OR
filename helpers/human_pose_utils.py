import numpy as np
import open3d as o3d
from helpers.linemesh import LineMesh


def human_pose_to_joints(p, limbs, radius=10):
    joint_points = o3d.utility.Vector3dVector(p)
    joint_colors = [[1, 0, 0] for _ in range(len(limbs))]
    joint_lines = o3d.utility.Vector2iVector(limbs)
    line_mesh = LineMesh(joint_points, joint_lines, joint_colors, radius=radius)
    line_mesh_geom = line_mesh.cylinder_segments
    joint_pc = o3d.geometry.PointCloud()
    points = []
    for elem in line_mesh_geom:
        points.append(np.asarray(elem.vertices))

    joint_pc.points = o3d.utility.Vector3dVector(np.concatenate(points))

    return joint_pc
