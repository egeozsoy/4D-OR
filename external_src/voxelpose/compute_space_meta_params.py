import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform.rotation import Rotation


def load_full_T(root_path: Path, cam_count=6):
    full_T = []
    for c_idx in range(1, cam_count + 1):
        cam_json_path = root_path / f'camera0{c_idx}.json'
        with cam_json_path.open() as f:
            cam_info = json.load(f)['value0']

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
            full_T.append(extrinsics[:3, 3])

    return np.stack(full_T)


if __name__ == '__main__':
    full_T = load_full_T(Path('../../datasets/4D-OR/export_holistic_take1_processed')) * 500
    space_sizes = np.abs(full_T.max(0)) + np.abs(full_T.min(0))
    space_centers = (full_T.max(0) + full_T.min(0)) / 2

    print('\nSPACE SIZE:')
    for space_size in space_sizes:
        print(f'{space_size:.1f}')

    print('\nSPACE CENTER:')
    for space_center in space_centers:
        print(f'{space_center:.1f}')
