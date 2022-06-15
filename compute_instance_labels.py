import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from scipy.spatial.distance import cdist
from tqdm.contrib.concurrent import process_map  # or thread_map

from external_src.group_free_3D.pointnet2.pointnet2_utils import furthest_point_sample
from helpers.configurations import OBJECT_POSES_PATH, INSTANCE_LABELS_PATH, INSTANCE_LABELS_PRED_PATH, OBJECT_LABEL_MAP, \
    LIMBS, IDX_TO_BODY_PART, POSE_PREDICTION_PATH, GROUP_FREE_PREDICTIONS_PATH, STATIONARY_OBJECTS
from helpers.human_pose_utils import human_pose_to_joints
from helpers.object_pose_utils import get_object_poses
from helpers.utils import coord_transform_human_pose_tool_to_OR_4D

random.seed(1)

NPOINTS = 200
CLOSENESS_THRESHOLD = 75
# seing more context around humans is not bad
object_poses_path = OBJECT_POSES_PATH / 'vs_0.01_rf_0.25_maxnn_500_ft_0.25'
FROM_GT = False


def vec_ang(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)


def heading2rotmat(heading_angle):
    rotmat = np.zeros((3, 3))
    rotmat[1, 1] = 1
    rotmat[2, 2] = 1
    cosval = np.cos(heading_angle)
    sinval = np.sin(heading_angle)
    rotmat[0, 0] = cosval
    rotmat[0, 2] = sinval
    rotmat[2, 0] = -sinval
    rotmat[2, 2] = cosval
    return rotmat


def convert_oriented_box_to_pose(box):
    ctr = box[:3]
    lengths = box[3:6]
    trns = np.eye(4)
    trns[0:3, 3] = ctr
    trns[3, 3] = 1.0
    trns[0:3, 0:3] = heading2rotmat(box[6])
    # Create grid
    grid1Dx = np.linspace(-lengths[0] / 2, lengths[0] / 2, 20)
    grid1Dy = np.linspace(-lengths[1] / 2, lengths[1] / 2, 20)
    grid1Dz = np.linspace(-lengths[2] / 2, lengths[2] / 2, 20)
    gridx, gridy, gridz = np.meshgrid(grid1Dx, grid1Dy, grid1Dz)
    box_pose = o3d.geometry.PointCloud()
    box_pose.points = o3d.utility.Vector3dVector(np.concatenate([gridx.reshape(-1, 1), gridy.reshape(-1, 1), gridz.reshape(-1, 1)], axis=1))
    box_pose.transform(trns)

    return box_pose


def compute_human_instance_labels(human_pose, point_cloud, exception_for_hands=False):
    head = human_pose[0]
    feet_center = np.mean([human_pose[12], human_pose[13]], axis=0)
    main_axis = np.abs(head - feet_center).argmax()  # Calculated by looking at which axis has the biggest change
    positive_direction = (head - feet_center)[main_axis] > 0
    joint_pc = human_pose_to_joints(human_pose, LIMBS, radius=30)
    xmin, ymin, zmin = np.asarray(joint_pc.points).min(0) - 100
    xmax, ymax, zmax = np.asarray(joint_pc.points).max(0) + 100
    if main_axis == 0:  # Purpose is to leave extra place above the head
        if positive_direction:
            xmax += 100
        else:
            xmin += -100
    elif main_axis == 1:
        if positive_direction:
            ymax += 100
        else:
            ymin += -100
    elif main_axis == 2:
        if positive_direction:
            zmax += 100
        else:
            zmin += -100
    pc_points = np.asarray(point_cloud.points)
    in_bbox_mask = (pc_points[:, 0] >= xmin) * (pc_points[:, 0] <= xmax) * (pc_points[:, 1] >= ymin) * (pc_points[:, 1] <= ymax) * (
            pc_points[:, 2] >= zmin) * (pc_points[:, 2] <= zmax)
    bbox_points = pc_points[in_bbox_mask]
    object_points = np.asarray(joint_pc.points)
    sample = furthest_point_sample(torch.from_numpy(object_points).unsqueeze(0).cuda().float(), NPOINTS)[0].cpu()
    object_points = object_points[sample]
    if positive_direction:
        edge_case_point = max(object_points, key=lambda x: x[main_axis])
        edge_case_point[main_axis] += 100  # virtual point higher in head axis
    else:
        edge_case_point = min(object_points, key=lambda x: x[main_axis])
        edge_case_point[main_axis] += -100  # virtual point lower in head axis
    object_points = np.concatenate([object_points, edge_case_point[None]])
    dst_matrix = cdist(bbox_points, object_points)
    close_mask = dst_matrix.min(1) < CLOSENESS_THRESHOLD
    full_mask = np.array([i for i in range(len(pc_points))])[in_bbox_mask][close_mask]

    if exception_for_hands:
        leftwrist = human_pose[IDX_TO_BODY_PART.index('leftwrist')]
        rightwrist = human_pose[IDX_TO_BODY_PART.index('rightwrist')]
        hand_points = np.stack([leftwrist, rightwrist])
        xmin, ymin, zmin = hand_points.min(0) - 100
        xmax, ymax, zmax = hand_points.max(0) + 100
        in_bbox_mask = (pc_points[:, 0] >= xmin) * (pc_points[:, 0] <= xmax) * (pc_points[:, 1] >= ymin) * (pc_points[:, 1] <= ymax) * (
                pc_points[:, 2] >= zmin) * (pc_points[:, 2] <= zmax)
        bbox_points = pc_points[in_bbox_mask]
        dst_matrix = cdist(bbox_points, object_points)
        close_mask = dst_matrix.min(1) < (CLOSENESS_THRESHOLD * 2)
        hand_mask = np.array([i for i in range(len(pc_points))])[in_bbox_mask][close_mask]
        return full_mask, hand_mask

    return full_mask


def _process_take_helper(take_idx):
    pcd_paths = sorted(list(Path(f'datasets/4D-OR/export_holistic_take{take_idx}_processed/pcds').glob('*.pcd')))
    export_human_name_to_3D_joints_path = Path(f'datasets/4D-OR/human_name_to_3D_joints') / f'{take_idx}_GT_{FROM_GT}.npz'
    print(export_human_name_to_3D_joints_path)
    all_human_name_to_3D_joints = {}
    print(f'Using GT {FROM_GT}')
    for pcd_path in pcd_paths:
        print(pcd_path)
        pcd_idx_str = pcd_path.name.replace('.pcd', '')
        point_cloud = o3d.io.read_point_cloud(str(pcd_path))
        instance_labels = np.zeros(len(point_cloud.points), dtype=np.int8) - 1

        # Infer object instance labels
        if FROM_GT:
            # Get objects
            pcd_objects_path = object_poses_path / f'{take_idx}_{pcd_path.name.replace(".pcd", "")}.npz'
            stationary_objects_path = object_poses_path / f'{take_idx}_stationary_objects.npz'
            json_path = object_poses_path / f'{take_idx}_{pcd_path.name.replace(".pcd", "")}_manual.json'
            registered_objects = np.load(str(pcd_objects_path), allow_pickle=True)['arr_0'].item()
            stationary_objects = {k: v for k, v in np.load(str(stationary_objects_path), allow_pickle=True)['arr_0']}
            if pcd_path.name > '000198' and int(take_idx) == 10:
                stationary_objects['datasets/4D-OR/object_scans/secondary_table/10.ply'][:3, 3] += [-0.05, 0, -0.05]
            registered_objects = {k: v for k, v in registered_objects.items() if k.split("/")[3] not in STATIONARY_OBJECTS}
            registered_objects = {**registered_objects, **stationary_objects}  # merge dicts
            with json_path.open() as f:
                corresponding_json = json.load(f)
            object_poses, object_names = get_object_poses(registered_objects)
            objects = []
            for object_pose, object_name in zip(object_poses, object_names):
                if object_name in corresponding_json['false_objects']:
                    continue
                objects.append((object_name, deepcopy(object_pose)))
        else:
            predicted_objects_path = GROUP_FREE_PREDICTIONS_PATH / f'{take_idx}_{pcd_idx_str}.npz'
            predictions = np.load(str(predicted_objects_path), allow_pickle=True)['arr_0'].item()
            classes = predictions['classes_nms']
            preds = predictions['bboxes_nms']
            scores = predictions['scores_nms']
            # Limit to one box per class
            unique_classes = np.unique(classes)
            filtered_preds = []
            filtered_classes = []
            for unique_class in unique_classes:
                class_mask = classes == unique_class
                max_idx = scores[class_mask].argmax()
                filtered_preds.append(preds[class_mask][max_idx])
                filtered_classes.append(unique_class)

            classes = np.asarray(filtered_classes)
            preds = np.asarray(filtered_preds)
            preds[:, :6] *= 1000

            objects = []
            for pred, cls in zip(preds, classes):
                name = {v: k for k, v in OBJECT_LABEL_MAP.items()}[cls]
                if name in ['operating_table', 'anesthesia_equipment']:  # Eventhough -ang makes more sense, for most objects ang works better
                    pred[6] *= -1
                object_pose = convert_oriented_box_to_pose(pred)
                cls_to_object_name_map = {v: k for k, v in OBJECT_LABEL_MAP.items()}
                objects.append((cls_to_object_name_map[cls], deepcopy(object_pose)))

        for idy, (object_name, object_pose) in enumerate(objects):
            xmin, ymin, zmin = np.asarray(object_pose.points).min(0)
            xmax, ymax, zmax = np.asarray(object_pose.points).max(0)
            pc_points = np.asarray(point_cloud.points)
            in_bbox_mask = (pc_points[:, 0] >= xmin) * (pc_points[:, 0] <= xmax) * (pc_points[:, 1] >= ymin) * (pc_points[:, 1] <= ymax) * (
                    pc_points[:, 2] >= zmin) * (pc_points[:, 2] <= zmax)
            bbox_points = pc_points[in_bbox_mask]
            object_points = np.asarray(object_pose.points)
            # sample = np.random.choice(len(object_points), 1000, replace=False)
            sample = furthest_point_sample(torch.from_numpy(object_points).unsqueeze(0).cuda().float(), NPOINTS)[0].cpu()
            object_points = object_points[sample]
            dst_matrix = cdist(bbox_points, object_points)
            close_mask = dst_matrix.min(1) < CLOSENESS_THRESHOLD
            full_mask = np.array([i for i in range(len(pc_points))])[in_bbox_mask][close_mask]
            instance_labels[full_mask] = OBJECT_LABEL_MAP[object_name]

        # Infer human instance labels
        if FROM_GT:
            human_pose_json_path = Path(f'datasets/4D-OR/export_holistic_take{take_idx}_processed/annotations') / f'{pcd_idx_str}.json'
            human_name_to_3D_joints = {}
            if human_pose_json_path.exists():
                with human_pose_json_path.open() as f:
                    human_pose_json = json.load(f)

                human_names = sorted({elem['humanName'] for elem in human_pose_json['labels']})
                h_idx = 0
                for human_name in human_names:
                    human_joints = [elem for elem in human_pose_json['labels'] if elem['humanName'] == human_name]
                    human_pose = []
                    joint_positions = {}
                    for human_joint in human_joints:
                        joint_positions[human_joint['jointName']] = (
                            human_joint['point3d']['location']['x'], human_joint['point3d']['location']['y'], human_joint['point3d']['location']['z'])

                    for body_part in IDX_TO_BODY_PART:
                        human_pose.append(joint_positions[body_part])

                    human_pose = np.asarray(human_pose)
                    human_pose = coord_transform_human_pose_tool_to_OR_4D(human_pose)  # Here we are working in the x500 space
                    if human_name == 'Patient':
                        h_name = 'Patient'
                    else:
                        h_name = f'human_{h_idx}'
                        h_idx += 1
                    human_name_to_3D_joints[h_name] = human_pose
                    full_mask, hand_mask = compute_human_instance_labels(human_pose, point_cloud, exception_for_hands=True)
                    instance_labels[full_mask] = OBJECT_LABEL_MAP[h_name]
                    unlabeled_mask = instance_labels[hand_mask] == -1
                    instrument_table_mask = instance_labels[hand_mask] == OBJECT_LABEL_MAP['instrument_table']
                    secondary_table_mask = instance_labels[hand_mask] == OBJECT_LABEL_MAP['secondary_table']
                    operating_table_mask = instance_labels[hand_mask] == OBJECT_LABEL_MAP['operating_table']
                    instance_labels[hand_mask[
                        np.logical_or(np.logical_or(np.logical_or(unlabeled_mask, instrument_table_mask), secondary_table_mask), operating_table_mask)]] = \
                        OBJECT_LABEL_MAP[h_name]

        else:
            predicted_pose_path = POSE_PREDICTION_PATH / f"pred_{take_idx}_{pcd_idx_str}.npy"
            human_name_to_3D_joints = {}
            if predicted_pose_path.exists():
                human_poses = np.load(str(predicted_pose_path))
                for h_idx, human_pose in enumerate(human_poses):
                    h_name = f'human_{h_idx}'  # Here we won't know the patient from others
                    human_name_to_3D_joints[h_name] = human_pose
                    full_mask, hand_mask = compute_human_instance_labels(human_pose, point_cloud, exception_for_hands=True)
                    instance_labels[full_mask] = OBJECT_LABEL_MAP[h_name]
                    unlabeled_mask = instance_labels[hand_mask] == -1
                    instrument_table_mask = instance_labels[hand_mask] == OBJECT_LABEL_MAP['instrument_table']
                    secondary_table_mask = instance_labels[hand_mask] == OBJECT_LABEL_MAP['secondary_table']
                    operating_table_mask = instance_labels[hand_mask] == OBJECT_LABEL_MAP['operating_table']
                    instance_labels[hand_mask[
                        np.logical_or(np.logical_or(np.logical_or(unlabeled_mask, instrument_table_mask), secondary_table_mask), operating_table_mask)]] = \
                        OBJECT_LABEL_MAP[h_name]

        if FROM_GT:
            instance_label_path = INSTANCE_LABELS_PATH / f'{take_idx}_{pcd_path.name.replace(".pcd", "")}.npz'
        else:
            instance_label_path = INSTANCE_LABELS_PRED_PATH / f'{take_idx}_{pcd_path.name.replace(".pcd", "")}.npz'
        np.savez_compressed(str(instance_label_path), instance_labels)
        all_human_name_to_3D_joints[f'{pcd_idx_str}'] = human_name_to_3D_joints

    np.savez_compressed(str(export_human_name_to_3D_joints_path), all_human_name_to_3D_joints)


# human key ordering here should be the same as for the labeling, else for scene graph prediction, we will have non sensicaly labels
def main():
    process_map(_process_take_helper, range(1, 11), max_workers=6)


if __name__ == '__main__':
    main()
