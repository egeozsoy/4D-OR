# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from pathlib import Path

import cv2
import json_tricks as json
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

TAKE_SPLIT = {'train': [1, 3, 5, 7, 9, 10], 'val': [4, 8], 'test': [2, 6]}


class OR_4DDataset(Dataset):

    def __init__(self, root, dataset, data_format, transform=None,
                 target_transform=None):
        self.name = 'OR_4D'
        self.root = root
        self.dataset = dataset
        self.data_format = data_format
        self.transform = transform
        self.target_transform = target_transform
        # self.inference = False
        self.classes = ['__background__'] + ['person']
        logger.info('=> classes: {}'.format(self.classes))

        self.take_indices = TAKE_SPLIT[self.dataset]

        print(f'{dataset} Using Takes {self.take_indices}')

        self.num_classes = len(self.classes)
        self.annotations = self.get_annotations()
        self.image_dicts = self.get_image_dicts()

    def get_annotations(self):
        take_to_annotations = {}
        for take_idx in self.take_indices:
            annotations_path = Path(f'../../datasets/4D-OR/export_holistic_take{take_idx}_processed/2D_keypoint_annotations.json')
            with annotations_path.open() as f:
                annotations = json.load(f)
                take_to_annotations[take_idx] = annotations

        return take_to_annotations

    def get_image_dicts(self):
        image_dicts = []
        image_id_counter = 0
        for take_idx in self.take_indices:
            with open(f'../../datasets/4D-OR/export_holistic_take{take_idx}_processed/timestamp_to_pcd_and_frames_list.json') as f:
                timestamp_to_pcd_and_frames_list = json.load(f)

            for idx, (_, corresponding_channels) in enumerate(timestamp_to_pcd_and_frames_list):
                for c_idx in range(1, 7):
                    rgb_str = corresponding_channels[f'color_{c_idx}']
                    annotation = self.annotations[take_idx][f'{str(idx).zfill(6)}_{c_idx}']
                    image_name = f'camera0{c_idx}_colorimage-{rgb_str}.jpg'
                    image_path = f'../../datasets/4D-OR/export_holistic_take{take_idx}_processed/colorimage/{image_name}'
                    image_dict = {'take_idx': take_idx, 'cam': c_idx, 'image_name': image_name, 'annotations': annotation, 'image_id': image_id_counter,
                                  'image_path': image_path}
                    if len([elem for elem in image_dicts if elem['image_path'] == image_path]) > 0:  # Make sure the image is not already included
                        continue
                    image_id_counter += 1
                    image_dicts.append(image_dict)
        return image_dicts

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        image_dict = self.image_dicts[index]
        img = cv2.imread(image_dict['image_path'],
                         cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
                         )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            raise Exception("Not Implemented Correctly")
            img = self.transform(img)

        return (img, image_dict)

    def __len__(self):
        return len(self.image_dicts)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def processKeypoints(self, keypoints):
        tmp = keypoints.copy()
        if keypoints[:, 2].max() > 0:
            p = keypoints[keypoints[:, 2] > 0][:, :2].mean(axis=0)
            num_keypoints = keypoints.shape[0]
            for i in range(num_keypoints):
                tmp[i][0:3] = [
                    float(keypoints[i][0]),
                    float(keypoints[i][1]),
                    float(keypoints[i][2])
                ]

        return tmp
