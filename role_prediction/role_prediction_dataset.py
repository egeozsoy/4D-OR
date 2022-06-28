import os.path as osp

import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
from pathlib import Path
from helpers.configurations import TAKE_SPLIT, OR_4D_DATA_ROOT_PATH
from role_prediction.graphormer.role_prediction_configs import USE_GT, USE_IMAGES
import json
import numpy as np
from pyvis.network import Network
from copy import deepcopy
import pickle

VISUALIZE = False
print(f'VISUALIZE: {VISUALIZE}')


def load_gt_scene_graphs_in_prediction_format():
    all_scan_relations = {}
    for take_idx in TAKE_SPLIT['train'] + TAKE_SPLIT['val'] + TAKE_SPLIT['test']:
        if take_idx in TAKE_SPLIT['train']:
            gt_rels_path = Path('data/relationships_train.json')
        elif take_idx in TAKE_SPLIT['val']:
            gt_rels_path = Path('data/relationships_validation.json')
        elif take_idx in TAKE_SPLIT['test']:
            gt_rels_path = Path(
                'data/relationships_test_dummy.json')  # TODO if you have access to relationships_test.json, you can use that instead
        else:
            raise Exception()
        with open(gt_rels_path) as f:
            all_scans_gt_rels = json.load(f)['scans']
        take_gt_rels = [scan_gt_rels for scan_gt_rels in all_scans_gt_rels if scan_gt_rels['take_idx'] == take_idx]
        take_gt_rels = sorted(take_gt_rels, key=lambda x: x['scan'])
        if len(take_gt_rels) == 0:
            continue
        for scan_gt_rels in take_gt_rels:
            object_idx_to_name = scan_gt_rels['objects']
            if 'Patient' in object_idx_to_name.values():
                humans = sorted(name for name in object_idx_to_name.values() if 'human' in name)
                if len(humans) == 0:
                    last_human_idx = -1
                else:
                    last_human_idx = int(humans[-1].split('_')[-1])
                patient_key = list(object_idx_to_name.keys())[list(object_idx_to_name.values()).index('Patient')]
                object_idx_to_name[patient_key] = f'human_{last_human_idx + 1}'
            rels = []
            for sub_idx, obj_idx, rel_idx, rel_name in scan_gt_rels['relationships']:
                rels.append((object_idx_to_name[str(sub_idx)], rel_name, object_idx_to_name[str(obj_idx)]))
            all_scan_relations[f'{scan_gt_rels["take_idx"]}_{scan_gt_rels["scan"]}'] = rels

    return all_scan_relations


def load_gt_role_labels(take_indices):
    take_frame_to_human_idx_to_name_and_joints = {}
    for take_idx in take_indices:
        root_path = OR_4D_DATA_ROOT_PATH / 'human_name_to_3D_joints'
        GT_take_human_name_to_3D_joints = np.load(str(root_path / f'{take_idx}_GT_True.npz'), allow_pickle=True)['arr_0'].item()
        if take_idx in TAKE_SPLIT['train']:
            gt_rels_path = Path('data/relationships_train.json')
        elif take_idx in TAKE_SPLIT['val']:
            gt_rels_path = Path('data/relationships_validation.json')
        elif take_idx in TAKE_SPLIT['test']:
            gt_rels_path = Path(
                'data/relationships_test_dummy.json')  # TODO if you have access to relationships_test.json, you can use that instead
        else:
            raise Exception()
        with open(gt_rels_path) as f:
            all_scans_gt_rels = json.load(f)['scans']
        for scan_gt_rel in all_scans_gt_rels:
            if scan_gt_rel['take_idx'] != take_idx:
                continue
            if 'Patient' in scan_gt_rel['objects'].values():
                scan_gt_rel['human_idx_to_name']['Patient'] = 'Patient'
            take_frame_str = f'{take_idx}_{scan_gt_rel["scan"]}'
            human_indices = list(scan_gt_rel['human_idx_to_name'].keys())
            human_idx_to_human_name_and_joints = {}
            for human_idx in human_indices:
                try:
                    name = scan_gt_rel['human_idx_to_name'][human_idx]
                    joints = GT_take_human_name_to_3D_joints[scan_gt_rel["scan"]][human_idx]
                    human_idx_to_human_name_and_joints[human_idx] = (name, joints)
                except Exception as e:
                    continue

            take_frame_to_human_idx_to_name_and_joints[take_frame_str] = human_idx_to_human_name_and_joints

    return take_frame_to_human_idx_to_name_and_joints


class RolePredictionDataset(Dataset):
    def __init__(self, split='train', transform=None, pre_transform=None):
        '''
        First method: Role Prediction based on tracks, use most common label in track to label tracks. Before evaluation, apply the same algorithm as for the rule based algorithm.
        '''
        root = f'role_prediction/role_prediction_dataset_GT_{USE_GT}_IMAGES_{USE_IMAGES}'
        self.all_scan_names = []
        if USE_GT:
            self.all_scan_relations = load_gt_scene_graphs_in_prediction_format()
        else:
            split_name = split if split != 'val' else 'validation'
            file_name = f'scan_relations_training_no_gt_{split_name}_scans.json' if not USE_IMAGES else f'scan_relations_training_no_gt_images_{split_name}_scans.json'
            with open(file_name) as f:
                self.all_scan_relations = json.load(f)
                self.all_scan_relations = {k.rsplit('_', 1)[0]: v for k, v in self.all_scan_relations.items()}
        self.take_indices = TAKE_SPLIT[split]
        self.split = split
        print(f'{split} Using Takes {self.take_indices}')
        # Load labels per scene graph, which we will use to label the tracks
        self.GT_take_frame_to_human_idx_to_name_and_joints = load_gt_role_labels(self.take_indices)

        self.take_to_scan_indices = {}

        for take_idx in self.take_indices:
            scan_names = sorted(list(Path(f'datasets/4D-OR/export_holistic_take{take_idx}_processed/pcds').glob('*.pcd')))
            scan_names = [f'{take_idx}_{elem.name.replace(".pcd", "")}' for elem in scan_names]
            self.take_to_scan_indices[take_idx] = scan_names
        super().__init__(root, transform, pre_transform)

    def objname_to_index(self, objname):
        obj_name_to_index = {
            'anesthesia_equipment': 1,
            'operating_table': 2,
            'instrument_table': 3,
            'secondary_table': 4,
            'instrument': 5,
            'object': 6,
            'human': 7,
            'TARGET': 8,
            'assisting': 9,
            'cementing': 10,
            'cleaning': 11,
            'closeto': 12,
            'cutting': 13,
            'drilling': 14,
            'hammering': 15,
            'holding': 16,
            'lyingon': 17,
            'operating': 18,
            'preparing': 19,
            'sawing': 20,
            'suturing': 21,
            'touching': 22
        }
        if 'human' in objname or 'Patient' in objname:  # We don't care about patient human_0 human_1 etc. everything is human (We don't seperate patient here, because voxelpose also won't seperate it)
            objname = 'human'
        elif '$' in objname:
            objname = objname.split('_')[1].lower()

        return obj_name_to_index[objname]

    def role_to_index(self, role):
        role_to_index = {
            'Patient': 0,
            'head-surgeon': 1,
            'assistant-surgeon': 2,
            'circulating-nurse': 3,
            'anaesthetist': 4,
        }
        return role_to_index[role]

    @property
    def processed_file_names(self):
        return sorted([elem.name for elem in Path(self.processed_dir).glob(f'data_{self.split}*.pt')])

    def process(self):
        role_occ_count = np.zeros(5)
        for take_idx in self.take_to_scan_indices.keys():
            with open(f'datasets/4D-OR/human_name_to_3D_joints/{take_idx}_scene_graph_track_GT_{USE_GT}.pickle', 'rb') as f:
                tracks = pickle.load(f)

            for track_idx, track in enumerate(tracks):
                all_corresponding_scene_graphs = []
                target_human_indices_joints = []
                # Collect all scene graphs matching to this target, rename target node to human_target
                for frame_str, (h_name, h_joint) in sorted(track['timestamp_to_human_pose'].items()):
                    matching_scene_graph = deepcopy(
                        [[key, [list(r) for r in scan_rel]] for key, scan_rel in self.all_scan_relations.items() if key == f'{take_idx}_{frame_str}'])[0]
                    for rel in matching_scene_graph[1]:
                        if rel[0] == h_name:
                            rel[0] = 'TARGET'
                        elif rel[2] == h_name:
                            rel[2] = 'TARGET'
                    all_corresponding_scene_graphs.append(matching_scene_graph)
                    target_human_indices_joints.append((h_name, h_joint))

                role_labels = []
                scene_graph_datas = []
                for scan_relations, target_human_idx_joints in zip(all_corresponding_scene_graphs, target_human_indices_joints):
                    corresponding_GT_humans = self.GT_take_frame_to_human_idx_to_name_and_joints[scan_relations[0]]
                    role_label = None
                    min_dist = 10000000
                    for human_idx, human_details in corresponding_GT_humans.items():
                        dist = np.linalg.norm(target_human_idx_joints[1] - human_details[1])
                        if dist < min_dist:
                            min_dist = dist
                            role_label = human_details[0]

                    role_labels.append(role_label)

                    nodes = set()
                    for rel_idx, (sub, rel, obj) in enumerate(scan_relations[1]):
                        nodes.add(sub)
                        nodes.add(obj)
                        # rel is also a node
                        nodes.add(f'$_{rel}_{rel_idx}')

                    nodes = sorted(nodes)
                    edges = []
                    for rel_idx, (sub, rel, obj) in enumerate(scan_relations[1]):
                        rel_full_name = f'$_{rel}_{rel_idx}'
                        edges.append((nodes.index(sub), nodes.index(rel_full_name), rel))
                        edges.append((nodes.index(rel_full_name), nodes.index(obj), rel))

                    if len(edges) == 0:  # Don't attempt to classify this track, it is not possible
                        continue

                    indices = [list((edge[0], edge[1])) for edge in edges]
                    node_features = torch.tensor([self.objname_to_index(objname) for objname in nodes], dtype=torch.long).unsqueeze(1)
                    edge_features = torch.tensor([1 for _ in edges], dtype=torch.long)
                    edge_index = torch.tensor(indices, dtype=torch.long)

                    data = Data(x=node_features, edge_index=edge_index.t().contiguous(), edge_attr=edge_features, edge_labels=[elem[2] for elem in edges],
                                is_target=torch.from_numpy(np.asarray(nodes) == 'TARGET'))
                    scene_graph_datas.append(data)

                agg_role_label = max(set(role_labels), key=role_labels.count)
                if agg_role_label == 'none' or agg_role_label is None:
                    continue
                label_tensor = torch.tensor([self.role_to_index(agg_role_label)], dtype=torch.long)
                if len(scene_graph_datas) == 0:
                    continue
                role_occ_count[label_tensor] += 1
                torch.save((scene_graph_datas, label_tensor, take_idx, track_idx),
                           osp.join(self.processed_dir, f'data_{self.split}_{take_idx}_{str(track_idx).zfill(3)}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, self.processed_file_names[idx]))
        return data


if __name__ == '__main__':
    dataset = RolePredictionDataset()
    a = dataset[0]
