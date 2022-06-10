import numpy as np
import open3d as o3d
from pathlib import Path

from helpers.configurations import INSTANCE_LABELS_PATH, INSTANCE_LABELS_PRED_PATH, OBJECT_LABEL_MAP, OBJECT_COLOR_MAP


def main():
    FROM_GT = False
    print(f'Using GT {FROM_GT}')

    for take_idx in range(1, 11):
        pcd_paths = sorted(list(Path(f'4D-OR/export_holistic_take{take_idx}_processed/pcds').glob('*.pcd')))

        for pcd_path in pcd_paths:
            point_cloud = o3d.io.read_point_cloud(str(pcd_path))
            point_cloud_colors = np.asarray(point_cloud.colors)
            if FROM_GT:
                instance_label_path = INSTANCE_LABELS_PATH / f'{take_idx}_{pcd_path.name.replace(".pcd", "")}.npz'
            else:
                instance_label_path = INSTANCE_LABELS_PRED_PATH / f'{take_idx}_{pcd_path.name.replace(".pcd", "")}.npz'
            instance_labels = np.load(str(instance_label_path))['arr_0']
            for obj_name, color in OBJECT_COLOR_MAP.items():
                label_id = OBJECT_LABEL_MAP[obj_name]
                mask = instance_labels == label_id
                point_cloud_colors[mask] = color

            o3d.visualization.draw_geometries([point_cloud], window_name=pcd_path.name)


if __name__ == '__main__':
    main()
