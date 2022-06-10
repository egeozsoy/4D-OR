import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation


def apply_data_augmentation_to_object_pcs(object_points, config=None):
    if config is None:
        config = {'brightness': 0.1, 'colors': 0.05, 'x_rot': 20, 'y_rot': 45, 'z_rot': 20, 'shift': 0.15,
                  'scale': (0.4, 1.6)}
    new_object_points = []
    for points in object_points:
        # Random brightness change
        points[:, 3:6] += np.random.uniform(-config['brightness'], config['brightness'])
        # Random hue change
        points[:, 3:6] += torch.FloatTensor(3).uniform_(-config['colors'], config['colors'])
        # Make sure rgb values are still between 0 and 1
        points[:, 3:6].clamp_(0, 1)

        # Shift the whole point cloud
        points[:, :3] += torch.FloatTensor(3).uniform_(-config['shift'], config['shift'])

        # Change center to origin
        current_pos = points[:, :3].mean(0)
        points[:, :3] -= current_pos

        y_rot = np.random.uniform(-config['y_rot'], config['y_rot'])
        x_rot = np.random.uniform(-config['x_rot'], config['x_rot'])
        z_rot = np.random.uniform(-config['z_rot'], config['z_rot'])

        points[:, :3] = torch.matmul(points[:, :3], torch.from_numpy(Rotation.from_euler('y', y_rot, degrees=True).as_matrix()).float())
        points[:, :3] = torch.matmul(points[:, :3], torch.from_numpy(Rotation.from_euler('y', x_rot, degrees=True).as_matrix()).float())
        points[:, :3] = torch.matmul(points[:, :3], torch.from_numpy(Rotation.from_euler('z', z_rot, degrees=True).as_matrix()).float())

        # Adjust scaling
        points[:, :3] *= np.random.uniform(config['scale'][0], config['scale'][1])
        # Translate back
        points[:, :3] += current_pos

        new_object_points.append(points)
    return torch.stack(new_object_points)


def apply_data_augmentations_to_relation_pcs(rel_points, rel_hand_points, gt_rels, relationNames):
    config = {'brightness': 0.1, 'colors': 0.025, 'x_rot': 10., 'y_rot': 20., 'z_rot': 10., 'shift': 0.1, 'scale': (0.4, 1.6),
              'hand_closeness_threshold': 0.2}
    # Only keep points close to the hands
    for rel_point, hand_point, gt_rel in zip(rel_points, rel_hand_points, gt_rels):
        rel_name = relationNames[gt_rel]
        if not rel_name in ['Cementing', 'Cleaning', 'Cutting', 'Drilling', 'Hammering', 'Sawing', 'Suturing', 'Touching']:
            continue  # apply on to specific relations
        thres = np.random.uniform(config['hand_closeness_threshold'], 1)
        mask = cdist(rel_point[:, :3], hand_point).min(1) > thres
        rel_point[mask] = torch.zeros(7)
    rel_points = apply_data_augmentation_to_object_pcs(rel_points)  # First treat it like an object and apply general stuff

    for rel_point in rel_points:
        # Apply to individual objects randomlly
        rel_point[rel_point[:, -1] == 1] = apply_data_augmentation_to_object_pcs([rel_point[rel_point[:, -1] == 1]], config)[0]
        rel_point[rel_point[:, -1] == 2] = apply_data_augmentation_to_object_pcs([rel_point[rel_point[:, -1] == 2]], config)[0]

    return rel_points
