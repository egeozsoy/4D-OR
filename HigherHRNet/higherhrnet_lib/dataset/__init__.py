# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# ------------------------------------------------------------------------------

from .build import make_dataloader
from .build import make_test_dataloader

OR_4D_part_labels = ['head', 'neck', 'left_shoulder', 'right_shoulder', 'left_hip',
                     'right_hip', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_knee', 'right_knee', 'left_foot', 'right_foot']

OR_4D_part_idx = {
    b: a for a, b in enumerate(OR_4D_part_labels)
}

OR_4D_part_orders = [
    ('head', 'neck'), ('neck', 'left_shoulder'), ('neck', 'right_shoulder'),
    ('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'), ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'), ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
    ('left_knee', 'left_hip'), ('right_knee', 'right_hip'), ('left_foot', 'left_knee'), ('right_foot', 'right_knee')
]

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

VIS_CONFIG = {
    'OR_4D':
        {
            'part_labels': OR_4D_part_labels,
            'part_idx': OR_4D_part_idx,
            'part_orders': OR_4D_part_orders
        },
}
