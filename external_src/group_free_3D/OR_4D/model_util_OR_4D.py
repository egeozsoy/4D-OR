# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np
import torch
from helpers.configurations import OBJECT_LABEL_MAP

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


class OR_4DDatasetConfig(object):
    def __init__(self):
        self.num_class = 4  # TODO adapt this if necessary
        self.num_heading_bin = 12
        self.num_size_cluster = 4

        # self.type2class = {'cabinet': 0, 'bed': 1, 'chair': 2, 'sofa': 3, 'table': 4, 'door': 5,
        #                    'window': 6, 'bookshelf': 7, 'picture': 8, 'counter': 9, 'desk': 10, 'curtain': 11,
        #                    'refrigerator': 12, 'showercurtrain': 13, 'toilet': 14, 'sink': 15, 'bathtub': 16,
        #                    'garbagebin': 17}
        self.id2class = OBJECT_LABEL_MAP
        self.class2type = {v: k for k, v in OBJECT_LABEL_MAP.items()}
        self.ids = np.array(sorted(list(OBJECT_LABEL_MAP.values())))
        self.mean_size_arr = np.load(os.path.join(ROOT_DIR, 'OR_4D/meta_data/OR_4D_means.npz'))['arr_0']
        self.type_mean_size = {}
        for i in range(self.num_size_cluster):
            self.type_mean_size[i] = self.mean_size_arr[i, :]

        self.class_frequencies = np.load('external_src/group_free_3D/OR_4D/meta_data/class_frequencies.npz', allow_pickle=True)['arr_0'].item()
        self.class_weights = {k: 1 / v for k, v in
                              self.class_frequencies.items()}  # If log weights are wanted -> {k: 1 / pow(v, 1 / 2) for k, v in self.class_frequencies.items()}
        self.class_weights = [self.class_weights[key] for key in sorted(self.class_weights.keys())]
        self.class_weights = torch.from_numpy(np.array(self.class_weights, dtype=np.float32))
        if torch.cuda.is_available():
            self.class_weights = self.class_weights.cuda()

    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from
            class center angle to current angle.

            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle
        '''
        num_class = self.num_heading_bin
        assert (angle >= -np.pi) and (angle <= np.pi)
        angle = angle % (2 * np.pi)
        assert (angle >= 0 and angle <= 2 * np.pi)
        angle_per_class = 2 * np.pi / float(num_class)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
        return class_id, residual_angle

    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class '''
        num_class = self.num_heading_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle > np.pi:
            angle = angle - 2 * np.pi
        return angle

    def class2size(self, pred_cls, residual):
        ''' Inverse function to size2class '''
        return self.mean_size_arr[pred_cls, :] + residual

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,), dtype=np.float32)
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle  # * -1
        return obb
