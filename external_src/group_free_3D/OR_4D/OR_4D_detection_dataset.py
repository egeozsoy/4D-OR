# coding: utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Dataset for object bounding box regression.
An axis aligned bounding box is parameterized by (cx,cy,cz) and (dx,dy,dz)
where (cx,cy,cz) is the center point of the box, dx is the x-axis length of the box.
"""
import json
import os
from copy import deepcopy

import numpy as np
import open3d as o3d
from scipy.spatial.transform.rotation import Rotation
from sklearn.decomposition import PCA
from torch.utils.data import Dataset

from external_src.group_free_3D.OR_4D.model_util_OR_4D import OR_4DDatasetConfig
from external_src.group_free_3D.OR_4D.OR_4D_utils import get_object_poses, vec_ang
from external_src.group_free_3D.utils import pc_util
from helpers.configurations import OBJECT_POSES_PATH, INSTANCE_LABELS_PATH, OBJECT_LABEL_MAP, \
    PREPROCESSED_RET_DICTS_PATH, OR_4D_DATA_ROOT_PATH, STATIONARY_OBJECTS, TAKE_SPLIT
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

DC = OR_4DDatasetConfig()
MAX_NUM_OBJ = 64
MEAN_COLOR_RGB = np.array([0.49, 0.54, 0.58])  # TODO adjust these if necessary


# MEAN_COLOR_RGB = np.array([0.218, 0.217, 0.248])


class OR_4DDetectionDataset(Dataset):

    def __init__(self, split_set='train', num_points=20000,
                 use_color=False, use_height=False, augment=False,
                 data_root=None):

        self.num_points = num_points
        self.use_color = use_color
        self.use_height = use_height
        self.augment = augment

        self.take_indices = TAKE_SPLIT[split_set]
        print(f'{split_set} Using Takes {self.take_indices}')

        all_scan_names = []

        for take_idx in self.take_indices:
            scan_names = sorted(list(Path(f'datasets/4D-OR/export_holistic_take{take_idx}_processed/pcds').glob('*.pcd')))
            scan_names = [f'{take_idx}_{elem.name.replace(".pcd", "")}' for elem in scan_names]
            all_scan_names.extend(scan_names)

        self.scan_names = sorted(all_scan_names)
        print(f'Using {len(self.scan_names)} scans')

    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            point_clouds: (N,3+C)
            center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
            sem_cls_label: (MAX_NUM_OBJ,) semantic class index
            heading_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
            heading_residual_label: (MAX_NUM_OBJ,)
            size_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
            size_residual_label: (MAX_NUM_OBJ,3)
            box_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
            point_obj_mask: (N,) with 0/1 with 1 indicating the point is in one of the object's OBB.
            point_instance_label: (N,) with int values in -1,...,num_box, indicating which object the point belongs to, -1 means a backgound point.
            scan_idx: int scan index in scan_names list
            pcl_color: unused
        """
        scan_name = self.scan_names[idx]
        save_name = PREPROCESSED_RET_DICTS_PATH / f'{scan_name}.npz'

        if save_name.exists():
            ret_dict = np.load(str(save_name), allow_pickle=True)['arr_0'].item()
            ret_dict['scan_name'] = scan_name
            return ret_dict
        else:
            take_idx, name = scan_name.split('_')
            pcd_path = (OR_4D_DATA_ROOT_PATH / f'export_holistic_take{take_idx}_processed/pcds') / f'{name}.pcd'
            object_poses_path = OBJECT_POSES_PATH / 'vs_0.01_rf_0.25_maxnn_500_ft_0.25'
            pcd = o3d.io.read_point_cloud(str(pcd_path))
            objects = []

            # Get objects
            pcd_objects_path = object_poses_path / f'{take_idx}_{pcd_path.name.replace(".pcd", "")}.npz'
            stationary_objects_path = object_poses_path / f'{take_idx}_stationary_objects.npz'
            json_path = object_poses_path / f'{take_idx}_{pcd_path.name.replace(".pcd", "")}_manual.json'
            registered_objects = {k.replace('INM', '4D-OR'): v for k, v in np.load(str(pcd_objects_path), allow_pickle=True)['arr_0'].item().items()}
            stationary_objects = {k.replace('INM', '4D-OR'): v for k, v in np.load(str(stationary_objects_path), allow_pickle=True)['arr_0']}
            if pcd_path.name > '000198' and int(take_idx) == 10:
                stationary_objects['datasets/4D-OR/object_scans/secondary_table/10.ply'][:3, 3] += [-0.05, 0, -0.05]
            registered_objects = {k: v for k, v in registered_objects.items() if k.split("/")[3] not in STATIONARY_OBJECTS}
            registered_objects = {**registered_objects, **stationary_objects}  # merge dicts
            with json_path.open() as f:
                corresponding_json = json.load(f)
            object_poses, object_names = get_object_poses(registered_objects)
            for object_pose, object_name in zip(object_poses, object_names):
                if object_name in corresponding_json['false_objects']:
                    continue
                objects.append((object_name, deepcopy(object_pose)))

            # Load instance(also semantic) labels
            instance_label_path = INSTANCE_LABELS_PATH / f'{take_idx}_{pcd_path.name.replace(".pcd", "")}.npz'
            instance_labels = np.load(str(instance_label_path))['arr_0']
            semantic_labels = np.copy(instance_labels)

            instance_bboxes = np.zeros((len(objects), 8))
            for idy, (object_name, object_pose) in enumerate(objects):
                xmin, ymin, zmin = np.asarray(object_pose.points).min(0)
                xmax, ymax, zmax = np.asarray(object_pose.points).max(0)

                center = np.asarray([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2])
                centered_obj_pc = np.asarray(object_pose.points) - center
                pca = PCA(n_components=1)
                pca.fit(centered_obj_pc[:, [0, 2]])  # only use X and Z
                ang = vec_ang(pca.components_[0], [1, 0])
                R = Rotation.from_euler('xyz', [0, ang, 0])
                centered_obj_pc = R.apply(centered_obj_pc)
                xmin = np.min(centered_obj_pc[:, 0])
                ymin = np.min(centered_obj_pc[:, 1])
                zmin = np.min(centered_obj_pc[:, 2])
                xmax = np.max(centered_obj_pc[:, 0])
                ymax = np.max(centered_obj_pc[:, 1])
                zmax = np.max(centered_obj_pc[:, 2])
                bbox = np.array([center[0], center[1], center[2], xmax - xmin, ymax - ymin, zmax - zmin, ang, OBJECT_LABEL_MAP[object_name]])
                instance_bboxes[idy, :] = bbox

            point_cloud = np.concatenate([np.asarray(pcd.points), np.asarray(pcd.colors)], axis=1)
            # scaling
            point_cloud[:, :3] /= 1000
            instance_bboxes[:, :6] /= 1000
            point_cloud[:, 3:] = (point_cloud[:, 3:] - MEAN_COLOR_RGB)

            if self.use_height:
                floor_height = np.percentile(point_cloud[:, 1], 0.99)
                height = point_cloud[:, 1] - floor_height
                point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)

                # ------------------------------- LABELS ------------------------------
            target_bboxes = np.zeros((MAX_NUM_OBJ, 8))
            target_bboxes_mask = np.zeros((MAX_NUM_OBJ))
            angle_classes = np.zeros((MAX_NUM_OBJ,))
            angle_residuals = np.zeros((MAX_NUM_OBJ,))
            size_classes = np.zeros((MAX_NUM_OBJ,))
            size_residuals = np.zeros((MAX_NUM_OBJ, 3))
            size_gts = np.zeros((MAX_NUM_OBJ, 3))

            point_cloud, choices = pc_util.random_sampling(point_cloud,
                                                           self.num_points, return_choices=True)

            instance_labels = instance_labels[choices]
            semantic_labels = semantic_labels[choices]

            target_bboxes_mask[0:instance_bboxes.shape[0]] = 1
            target_bboxes[0:instance_bboxes.shape[0], :] = instance_bboxes[:, 0:8]

            gt_centers = target_bboxes[:, 0:3]
            gt_centers[instance_bboxes.shape[0]:, :] += 1000.0  # padding centers with a large number
            point_obj_mask = np.zeros(self.num_points)
            point_instance_label = np.zeros(self.num_points) - 1
            for i_instance in np.unique(instance_labels):
                # find all points belong to that instance
                ind = np.where(instance_labels == i_instance)[0]
                # find the semantic label
                if semantic_labels[ind[0]] in DC.ids:
                    x = point_cloud[ind, :3]
                    center = 0.5 * (x.min(0) + x.max(0))
                    ilabel = np.argmin(((center - gt_centers) ** 2).sum(-1))
                    point_instance_label[ind] = ilabel
                    point_obj_mask[ind] = 1.0

            class_ind = [np.where(DC.ids == x)[0][0] for x in instance_bboxes[:, -1]]
            # NOTE: set size class as semantic class. Consider use size2class.
            size_classes[0:instance_bboxes.shape[0]] = class_ind
            size_residuals[0:instance_bboxes.shape[0], :] = target_bboxes[0:instance_bboxes.shape[0], 3:6] - DC.mean_size_arr[class_ind, :]
            size_gts[0:instance_bboxes.shape[0], :] = target_bboxes[0:instance_bboxes.shape[0], 3:6]

            for i in range(len(instance_bboxes)):
                angle_class, angle_residual = DC.angle2class(instance_bboxes[i][6])
                angle_classes[i] = angle_class
                angle_residuals[i] = angle_residual

            ret_dict = {}
            ret_dict['point_clouds'] = point_cloud.astype(np.float32)
            ret_dict['center_label'] = gt_centers.astype(np.float32)
            ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
            ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
            ret_dict['size_class_label'] = size_classes.astype(np.int64)
            ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
            ret_dict['size_gts'] = size_gts.astype(np.float32)
            target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
            target_bboxes_semcls[0:instance_bboxes.shape[0]] = [DC.ids[int(x)] for x in instance_bboxes[:, -1][0:instance_bboxes.shape[0]]]
            ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
            ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32)
            ret_dict['point_obj_mask'] = point_obj_mask.astype(np.int64)
            ret_dict['point_instance_label'] = point_instance_label.astype(np.int64)
            ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
            ret_dict['scan_name'] = scan_name

            np.savez_compressed(str(save_name), ret_dict)
            return ret_dict
