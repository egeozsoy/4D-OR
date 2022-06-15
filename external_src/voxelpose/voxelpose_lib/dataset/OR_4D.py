# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
from collections import defaultdict
from pathlib import Path

import copy
import json_tricks as json
import numpy as np
import os
import scipy.io as scio
from scipy.spatial.transform.rotation import Rotation

from voxelpose_lib.dataset.JointsDataset import JointsDataset
from voxelpose_lib.utils.transforms import projectPoints
import open3d as o3d

OR_4D_JOINTS_DEF = {
    'head': 0,
    'neck': 1,
    'left_shoulder': 2,
    'right_shoulder': 3,
    'left_hip': 4,
    'right_hip': 5,
    'left_elbow': 6,
    'right_elbow': 7,
    'left_wrist': 8,
    'right_wrist': 9,
    'left_knee': 10,
    'right_knee': 11,
    'leftfoot': 12,
    'rightfoot': 13
}

LIMBS = [
    [5, 4],  # (righthip-lefthip)
    [9, 7],  # (rightwrist - rightelbow)
    [7, 3],  # (rightelbow - rightshoulder)
    [2, 6],  # (leftshoulder - leftelbow)
    [6, 8],  # (leftelbow - leftwrist)
    [5, 3],  # (righthip - rightshoulder)
    [4, 2],  # (lefthip - leftshoulder)
    [3, 1],  # (rightshoulder - neck)
    [2, 1],  # (leftshoulder - neck)
    [1, 0],  # (neck - head)
    [10, 4],  # (leftknee,lefthip),
    [11, 5],  # (rightknee,righthip),
    [12, 10],  # (leftfoot,leftknee),
    [13, 11]  # (rightfoot,rightknee),

]

IDX_TO_BODY_PART = ['head', 'neck', 'leftshoulder', 'rightshoulder', 'lefthip', 'righthip', 'leftelbow', 'rightelbow', 'leftwrist', 'rightwrist', 'leftknee',
                    'rightknee', 'leftfoot', 'rightfoot']


def coord_transform_human_pose_tool_to_OR_4D(arr):
    # reverse of coord_transform_OR_4D_to_human_pose_tool
    arr *= 25
    arr[:, 2] += 1000
    arr[:, 1] *= -1
    arr = arr[:, [0, 2, 1]]
    return arr


TAKE_SPLIT = {'train': [1, 3, 5, 7, 9, 10], 'val': [4, 8], 'test': [2, 6]}


class OR_4D(JointsDataset):
    def __init__(self, cfg, image_set, is_train, transform=None, inference=False):
        self.pixel_std = 200.0
        self.joints_def = OR_4D_JOINTS_DEF
        super().__init__(cfg, image_set, is_train, transform)
        del self.scale_factor
        del self.rotation_factor
        del self.num_views
        self.take_indices = TAKE_SPLIT[image_set]
        print(f'{image_set} using indices: {self.take_indices}')

        self.take_to_annotations = {}
        self.take_to_timestamp_to_pcd_and_frames_list = {}

        for take_idx in self.take_indices:
            annotations_path = Path(f'../../datasets/4D-OR/export_holistic_take{take_idx}_processed/2D_keypoint_annotations.json')
            with annotations_path.open() as f:
                annotations = json.load(f)
                self.take_to_annotations[take_idx] = annotations

            with open(f'../../datasets/4D-OR/export_holistic_take{take_idx}_processed/timestamp_to_pcd_and_frames_list.json') as f:
                timestamp_to_pcd_and_frames_list = json.load(f)
                self.take_to_timestamp_to_pcd_and_frames_list[take_idx] = timestamp_to_pcd_and_frames_list

        self.limbs = LIMBS
        self.num_joints = len(OR_4D_JOINTS_DEF)
        self.cam_list = [1, 2, 3, 4, 5, 6]
        self.num_views = len(self.cam_list)

        self.pred_pose2d = self._get_pred_pose2d()
        self.inference = inference
        self.db = self._get_db()

        self.db_size = len(self.db)
        print(f'Using {image_set} imageset')

    def _get_pred_pose2d(self):
        pred_2d = np.load(f'../HigherHRNet/pred_or_4d_hrnet_coco_{self.image_set}.npz', allow_pickle=True)['arr_0'].item()
        return pred_2d

    def get_image_dicts(self):
        image_dicts = []
        image_id_counter = 0

        for take_idx in self.take_indices:
            for idx, (_, corresponding_channels) in enumerate(self.take_to_timestamp_to_pcd_and_frames_list[take_idx]):
                for c_idx in range(1, 7):
                    rgb_str = corresponding_channels[f'color_{c_idx}']
                    image_name = f'camera0{c_idx}_colorimage-{rgb_str}.jpg'
                    image_path = f'../../datasets/4D-OR/export_holistic_take{take_idx}_processed/colorimage/{image_name}'
                    image_dict = {'take_idx': take_idx, 'cam': c_idx, 'image_name': image_name, 'image_id': image_id_counter, 'image_path': image_path}
                    if len([elem for elem in image_dicts if elem['image_path'] == image_path]) > 0:  # Make sure the image is not already included
                        continue
                    image_id_counter += 1
                    image_dicts.append(image_dict)

        return image_dicts

    def transformGivenTrfMatrix(self, ann3d, tr_mat):
        """
        Transform the 3D points from one coordiate system to another given 4x4 transformation matrix
        :param ann3d: input 3D points
        :param tr_mat: 4x4 transformation matrix
        :return: transformed 3D points
        """
        X = ann3d[:, 0]
        Y = ann3d[:, 1]
        Z = ann3d[:, 2]
        pt3d = np.vstack((X, Y, Z, np.ones(len(X))))
        # transform points to room coordinate
        pt3d = np.dot(tr_mat, pt3d)
        pt3d = pt3d[0:3]
        return pt3d

    def projectCam3DTo2D(self, pose3D, K):
        """
        Project 3D point in camera coordinates to image given intrinsic camera parameters (focal-length and principal-point)
        :param pose3D: 3D points in camera coordinates
        :param camparam: intrinsic camera parameters
        :return: 2D point on the images
        """
        focal = K[0, 0], K[1, 1]
        pp = K[0, 2], K[1, 2]
        pose3D[2][pose3D[2] == 0.0] = 1.0  # replace zero with 1 to avoid divide by zeros
        p1 = ((np.divide(pose3D[0], pose3D[2])) * focal[0]) + pp[0]
        p2 = ((np.divide(pose3D[1], pose3D[2])) * focal[1]) + pp[1]
        return np.vstack((p1, p2))

    def _get_db(self):
        width = 2048
        height = 1536

        db = []
        cameras = self._get_cam(root_path=Path(f'../../datasets/4D-OR/export_holistic_take{self.take_indices[0]}_processed'))
        image_dicts = self.get_image_dicts()
        tmp_dict = {}
        for image_dict in image_dicts:
            tmp_dict[f'{image_dict["take_idx"]}_{image_dict["image_name"]}'] = image_dict

        image_dicts = tmp_dict

        for take_idx in self.take_indices:
            for idx, (_, corresponding_channels) in enumerate(self.take_to_timestamp_to_pcd_and_frames_list[take_idx]):
                pcd_idx_str = corresponding_channels['pcd']
                human_pose_json_path = Path(f'../../datasets/4D-OR/export_holistic_take{take_idx}_processed/annotations') / f'{pcd_idx_str}.json'
                bodies = []
                is_patient_mask = []
                if human_pose_json_path.exists():
                    with human_pose_json_path.open() as f:
                        human_pose_json = json.load(f)
                        human_names = {elem['humanName'] for elem in human_pose_json['labels']}
                        for _, human_name in enumerate(human_names):
                            human_pose = []
                            human_joints = [elem for elem in human_pose_json['labels'] if elem['humanName'] == human_name]
                            joint_positions = {}
                            for human_joint in human_joints:
                                joint_positions[human_joint['jointName']] = (
                                    human_joint['point3d']['location']['x'], human_joint['point3d']['location']['y'], human_joint['point3d']['location']['z'])

                            for body_part in IDX_TO_BODY_PART:
                                human_pose.append(joint_positions[body_part])
                            human_pose = np.asarray(human_pose)
                            human_pose = coord_transform_human_pose_tool_to_OR_4D(human_pose)
                            # human_pose /= 500
                            human_pose = np.stack(
                                [human_pose[:, 0], human_pose[:, 1], human_pose[:, 2], np.ones(len(human_pose)) + 1]).transpose().flatten().tolist()
                            bodies.append(human_pose)
                            is_patient_mask.append(human_name == 'Patient')
                if len(bodies) == 0 and not self.inference:
                    continue
                for k, cam in cameras.items():
                    color_image_idx_str = corresponding_channels[f'color_{k}']
                    identifier = f'{take_idx}_camera0{k}_colorimage-{color_image_idx_str}.jpg'
                    image_dict = image_dicts[identifier]
                    preds = self.pred_pose2d[identifier]
                    preds = [np.array(p) for p in
                             preds]

                    all_poses_3d = []
                    all_is_patient = []
                    all_poses_vis_3d = []
                    all_poses = []
                    all_poses_vis = []
                    for is_patient, body in zip(is_patient_mask, bodies):
                        pose3d = np.array(body).reshape((-1, 4))
                        pose3d = pose3d[:self.num_joints]

                        joints_vis = pose3d[:, -1] > 0.1

                        if not joints_vis[self.root_id[0]] or not joints_vis[self.root_id[1]]:
                            continue

                        all_poses_3d.append(pose3d[:, 0:3])
                        all_is_patient.append(is_patient)
                        all_poses_vis_3d.append(
                            np.repeat(
                                np.reshape(joints_vis, (-1, 1)), 3, axis=1))

                        pose3d_pc = o3d.geometry.PointCloud()
                        pose3d_pc.points = o3d.utility.Vector3dVector(pose3d[:, :3] / 500)
                        pose3d_pc.transform(np.linalg.inv(cam['extrinsics']))  # Bring from world to rgb camera coords
                        pose3d_pc.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # this is needed
                        obj_points = np.asarray(pose3d_pc.points)
                        # Project onto image
                        obj_points[:, 2][obj_points[:, 2] == 0.0] = 1.0  # replace zero with 1 to avoid divide by zeros
                        x = obj_points[:, 0]
                        y = obj_points[:, 1]
                        z = obj_points[:, 2]
                        u = (x * cam['fx'] / z) + cam['cx']
                        v = (y * cam['fy'] / z) + cam['cy']
                        pose2d = np.stack([u, v], axis=1)
                        x_check = np.bitwise_and(pose2d[:, 0] >= 0,
                                                 pose2d[:, 0] <= width - 1)
                        y_check = np.bitwise_and(pose2d[:, 1] >= 0,
                                                 pose2d[:, 1] <= height - 1)
                        check = np.bitwise_and(x_check, y_check)
                        joints_vis[np.logical_not(check)] = 0

                        all_poses.append(pose2d)
                        all_poses_vis.append(
                            np.repeat(
                                np.reshape(joints_vis, (-1, 1)), 2, axis=1))

                    if len(all_poses_3d) > 0 or self.inference:
                        our_cam = {}
                        our_cam['R'] = cam['R']
                        our_cam['T'] = cam['T']
                        our_cam['fx'] = cam['fx']
                        our_cam['fy'] = cam['fy']
                        our_cam['cx'] = cam['cx']
                        our_cam['cy'] = cam['cy']
                        our_cam['k'] = cam['distCoef'][[0, 1, 4]].reshape(3, 1)
                        our_cam['p'] = cam['distCoef'][[2, 3]].reshape(2, 1)
                        our_cam['extrinsics'] = cam['extrinsics']

                        db.append({
                            'key': f"{identifier}",
                            'image': image_dict['image_path'],
                            'joints_3d': all_poses_3d,
                            'is_patient_mask': all_is_patient,
                            'joints_3d_vis': all_poses_vis_3d,
                            'joints_2d': all_poses,
                            'joints_2d_vis': all_poses_vis,
                            'camera': our_cam,
                            'pred_pose2d': preds,
                            'pcd_idx_str': pcd_idx_str,
                            'take_idx': take_idx
                        })

        return db

    def _get_cam(self, root_path, cam_count=6):
        cameras = {}
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
                extrinsics = np.matmul(extrinsics, color2depth_transform)  # Extrinsics were given for the depth camera, convert them to color camera

                fov_x = cam_info['color_parameters']['fov_x']
                fov_y = cam_info['color_parameters']['fov_y']
                c_x = cam_info['color_parameters']['c_x']
                c_y = cam_info['color_parameters']['c_y']

                cameras[str(c_idx)] = {'K': intrinsics, 'distCoef': np.zeros(5, ), 'R': extrinsics[:3, :3], 'T': np.expand_dims(extrinsics[:3, 3], axis=1),
                                       'fx': np.asarray(fov_x), 'fy': np.asarray(fov_y), 'cx': np.asarray(c_x), 'cy': np.asarray(c_y), 'extrinsics': extrinsics}
        return cameras

    def __getitem__(self, idx):
        input, target_heatmap, target_weight, target_3d, meta, input_heatmap = [], [], [], [], [], []
        for k in range(self.num_views):
            i, th, tw, t3, m, ih = super().__getitem__(self.num_views * idx + k)
            input.append(i)
            target_heatmap.append(th)
            target_weight.append(tw)
            input_heatmap.append(ih)
            target_3d.append(t3)
            meta.append(m)

        return input, target_heatmap, target_weight, target_3d, meta, input_heatmap

    def __len__(self):
        return self.db_size // self.num_views

    def evaluate(self, preds, recall_threshold=500):
        total_gt = 0
        match_gt = 0
        correct_parts = np.zeros(10)
        total_parts = np.zeros(10)
        alpha = 0.5
        for idx in range(len(preds)):
            pred = preds[idx].copy()
            pred = pred[pred[:, 0, 3] >= 0, :, :3]
            input, target_heatmap, target_weight, target_3d, meta, input_heatmap = self[idx]
            gts = meta[0]['joints_3d']
            num_person = [elem['num_person'] for elem in meta]
            assert len(set(num_person)) == 1
            num_person = num_person[0]

            for person in range(num_person):
                gt = gts[person]
                if len(gt[0]) == 0:
                    continue
                if len(pred) == 0:
                    total_gt += 1
                    continue
                mpjpes = np.mean(np.sqrt(np.sum((gt[np.newaxis] - pred) ** 2, axis=-1)), axis=-1)
                min_n = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                if min_mpjpe < recall_threshold:
                    match_gt += 1
                total_gt += 1

                for j, k in enumerate(LIMBS):
                    total_parts[person] += 1
                    error_s = np.linalg.norm(pred[min_n, k[0], 0:3] - gt[k[0]])
                    error_e = np.linalg.norm(pred[min_n, k[1], 0:3] - gt[k[1]])
                    limb_length = np.linalg.norm(gt[k[0]] - gt[k[1]])
                    if (error_s + error_e) / 2.0 <= alpha * limb_length:
                        correct_parts[person] += 1

        actor_pcp = correct_parts / (total_parts + 1e-8)
        avg_pcp = np.mean(actor_pcp[:3])

        return actor_pcp, avg_pcp, None, match_gt / (total_gt + 1e-8)
