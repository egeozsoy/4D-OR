import json
from pathlib import Path
from helpers.scene_graph_vis_helpers import visualize_scene_graph
from tqdm import tqdm


def main():
    visualize_closeto = False
    USE_IMAGES = True
    print(f'Visualize CloseTo {visualize_closeto}')
    split_name = 'val'
    scene_graphs_path = Path(f'scan_relations_training_no_gt_{split_name}_scans.json') if not USE_IMAGES else Path(
        f'scan_relations_training_no_gt_images_{split_name}_scans.json')  # TODO adjust path as necessary

    suffix = ''
    if USE_IMAGES:
        suffix += '_using_images'
    if not visualize_closeto:
        suffix += '_no_closeto'

    save_root = Path(f'datasets/4D-OR/scene_graph_prediction_visualizations{suffix}')
    if not save_root.exists():
        save_root.mkdir()
    name_mapping = {'human_0': 'Human 0', 'human_1': 'Human 1', 'human_2': 'Human 2', 'human_3': 'Human 3', 'human_4': 'Human 4', 'human_5': 'Human 5',
                    'human_6': 'Human 6',
                    'secondary_table': 'Second Table', 'instrument_table': 'Instrument Table', 'anesthesia_equipment': 'Anesthesia Equipment',
                    'operating_table': 'Operating Table', 'Patient': 'Patient', 'instrument': 'Instrument'}
    with scene_graphs_path.open() as f:
        scans = json.load(f)

    for scan_id, relations in tqdm(scans.items()):
        take_idx, frame_number, _ = scan_id.split('_')
        if len(relations) == 0:
            continue
        rels = []
        for sub, rel, obj in relations:
            if not visualize_closeto and rel == 'CloseTo':
                continue
            if sub == 'object' or obj == 'object':
                continue
            rels.append((name_mapping[sub], name_mapping[obj], rel))

        visualize_scene_graph(rels, save_path=str(save_root / f'{take_idx}_{frame_number}.html'))


if __name__ == '__main__':
    main()
