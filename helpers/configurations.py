from pathlib import Path
from typing import Union

OBJECT_COLOR_MAP = {
    'anesthesia_equipment': (0.96, 0.576, 0.65),
    'operating_table': (0.2, 0.83, 0.72),
    'instrument_table': (0.93, 0.65, 0.93),
    'secondary_table': (0.90, 0.30, 0.63),
    'instrument': (1.0, 0.811, 0.129),
    'object': (0.61, 0.48, 0.04),
    'Patient': (0, 1., 0),
    'human_0': (1., 0., 0),
    'human_1': (0.9, 0., 0),
    'human_2': (0.85, 0., 0),
    'human_3': (0.8, 0., 0),
    'human_4': (0.75, 0., 0),
    'human_5': (0.7, 0., 0),
    'human_6': (0.65, 0., 0),
    'human_7': (0.6, 0., 0)
    # 'drill': (0.90, 0.30, 0.30),
    # 'saw': (1.0, 0.811, 0.129),
    # '': (0.90, 0.30, 0.30),
    # 'c-arm': (1.0, 0.811, 0.129),
    # 'c-arm_base': (0.61, 0.48, 0.04),
    # 'unidentified_1': (0.34, 0.65, 0.36),
    # 'unidentified_2': (0.22, 0.3, 0.83)
}

OBJECT_LABEL_MAP = {
    'anesthesia_equipment': 0,
    'operating_table': 1,
    'instrument_table': 2,
    'secondary_table': 3,
    'instrument': 4,
    'object': 5,
    'Patient': 9,
    'human_0': 10,
    'human_1': 11,
    'human_2': 12,
    'human_3': 13,
    'human_4': 14,
    'human_5': 15,
    'human_6': 16,
    'human_7': 17
}

TAKE_SPLIT = {'train': [1, 3, 5, 7, 9, 10], 'val': [4, 8], 'test': [2, 6]}

OR_4D_DATA_ROOT_PATH = Path('datasets/4D-OR')
EXPORT_HOLISTICS_PATHS = list(OR_4D_DATA_ROOT_PATH.glob('export_holistic_take*'))
OBJECT_POSES_PATH = OR_4D_DATA_ROOT_PATH / 'object_pose_results'
OBJECT_SCANS_PATH = OR_4D_DATA_ROOT_PATH / 'object_scans'
POSE_PREDICTION_PATH = Path('external_src/voxelpose/data/OR_4D_outputs')
PREPROCESSED_RET_DICTS_PATH = OR_4D_DATA_ROOT_PATH / Path('preprocessed_ret_dicts')
PREPROCESSED_RET_DICTS_PATH.mkdir(exist_ok=True)
INSTANCE_LABELS_PATH = OR_4D_DATA_ROOT_PATH / Path('instance_labels')
INSTANCE_LABELS_PATH.mkdir(exist_ok=True)
INSTANCE_LABELS_PRED_PATH = OR_4D_DATA_ROOT_PATH / Path('instance_labels_pred')
INSTANCE_LABELS_PRED_PATH.mkdir(exist_ok=True)
GROUP_FREE_PREDICTIONS_PATH = OR_4D_DATA_ROOT_PATH / Path('group_free_predictions')
GROUP_FREE_PREDICTIONS_PATH.mkdir(exist_ok=True)

DEPTH_SCALING = 2000

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

HUMAN_POSE_COLOR_MAP = {
    0: (255, 0, 0),
    1: (200, 0, 0),
    2: (68, 240, 65),
    3: (50, 166, 48),
    4: (65, 201, 224),
    5: (42, 130, 145),
    6: (66, 179, 245),
    7: (44, 119, 163),
    8: (245, 173, 66),
    9: (186, 131, 50)
}

IDX_TO_BODY_PART = ['head', 'neck', 'leftshoulder', 'rightshoulder', 'lefthip', 'righthip', 'leftelbow', 'rightelbow', 'leftwrist', 'rightwrist', 'leftknee',
                    'rightknee', 'leftfoot', 'rightfoot']

STATIONARY_OBJECTS = ['instrument_table',
                      'secondary_table']  # We might have to seperate these into different takes, if an object is only stationary in one take etc.
