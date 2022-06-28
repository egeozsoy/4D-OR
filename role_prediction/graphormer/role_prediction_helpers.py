from helpers.configurations import TAKE_SPLIT, OR_4D_DATA_ROOT_PATH
import numpy as np
from torch.utils.data import DataLoader
from functools import partial
from role_prediction.graphormer.data import get_dataset
from role_prediction.graphormer.collator import collator
import torch
from torch.nn import functional as F
import pickle
from pathlib import Path
import json
from copy import deepcopy
from sklearn.metrics import classification_report
from role_prediction.graphormer.role_prediction_configs import USE_GT, USE_IMAGES


def get_rels_path(take_idx, USE_GT, USE_IMAGES):
    if take_idx in TAKE_SPLIT['train']:
        if USE_GT:
            return Path('data/relationships_train.json')
        else:
            return Path(f'scan_relations_training_no_gt_train_scans.json') if not USE_IMAGES else Path(f'scan_relations_training_no_gt_images_train_scans.json')
    elif take_idx in TAKE_SPLIT['val']:
        if USE_GT:
            return Path('data/relationships_validation.json')
        else:
            return Path(
                f'scan_relations_training_no_gt_validation_scans.json') if not USE_IMAGES else Path(
                f'scan_relations_training_no_gt_images_validation_scans.json')
    elif take_idx in TAKE_SPLIT['test']:
        if USE_GT:
            return Path(
                'data/relationships_test_dummy.json')  # TODO if you have access to relationships_test.json, you can use that instead
        else:
            return Path(f'scan_relations_training_no_gt_test_scans.json') if not USE_IMAGES else Path(f'scan_relations_training_no_gt_images_test_scans.json')

    return None


def get_take_rels(rels_path, take_idx, USE_GT):
    if USE_GT:
        with open(rels_path) as f:
            all_scans_rels = json.load(f)['scans']
            take_rels = [scan_gt_rels for scan_gt_rels in all_scans_rels if scan_gt_rels['take_idx'] == take_idx]
    else:
        if not rels_path.exists():
            return None
        with open(rels_path) as f:
            all_scans_rels = json.load(f)
            all_scans_rels = {k.rsplit('_', 1)[0]: v for k, v in all_scans_rels.items()}
            take_rels = []
            for key, value in all_scans_rels.items():
                t_idx, scan_idx = key.split('_')
                t_idx = int(t_idx)
                if t_idx == take_idx:
                    take_rels.append({'take_idx': t_idx, 'scan': scan_idx, 'relationships': value})

    return take_rels


def compute_dist_matrix_between_human_nodes(gt_humans, pred_humans):
    dist_matrix = np.zeros((len(gt_humans), len(pred_humans)))
    for idx, gt_human in enumerate(gt_humans):
        for idy, pred_human in enumerate(pred_humans):
            # downsample for speed reasons
            dist = np.linalg.norm(gt_human[1] - pred_human[1][1])
            dist_matrix[idx, idy] = dist
    return dist_matrix


def match_human_preds_to_gt(gt_humans_to_joints, sg_humans_to_roles, sg_humans_to_joints):
    gt_humans_to_joints = sorted(gt_humans_to_joints.items())
    sg_humans_to_role_and_joints = {}
    for key in sg_humans_to_roles.keys():
        sg_humans_to_role_and_joints[key] = (sg_humans_to_roles[key], sg_humans_to_joints[key])
    sg_humans_to_role_and_joints = sorted(sg_humans_to_role_and_joints.items())
    renamed_sg_humans_to_roles = {}

    while len(gt_humans_to_joints) > 0 and len(sg_humans_to_role_and_joints) > 0:
        dist_matrix = compute_dist_matrix_between_human_nodes(gt_humans=gt_humans_to_joints, pred_humans=sg_humans_to_role_and_joints)
        min_value_index = np.unravel_index(dist_matrix.argmin(), dist_matrix.shape)
        gt_human_idx = min_value_index[0]
        pred_human_idx = min_value_index[1]
        renamed_sg_humans_to_roles[gt_humans_to_joints[gt_human_idx][0]] = sg_humans_to_role_and_joints[pred_human_idx][1][0]
        gt_humans_to_joints.pop(gt_human_idx)
        sg_humans_to_role_and_joints.pop(pred_human_idx)

    return renamed_sg_humans_to_roles


def infer_roles_in_sg(sg, take_tracks, take_track_to_score, take_idx):
    frame_str = sg['scan']
    track_indices_to_human = {}
    track_indices_to_guesses = {}
    sg_human_name_to_roles = {}
    sg_human_name_to_joints = {}
    for track_idx, track in enumerate(take_tracks):
        if frame_str in track['timestamp_to_human_pose']:
            track_indices_to_human[track_idx] = deepcopy(track['timestamp_to_human_pose'][frame_str])
            if f'{take_idx}_{track_idx}' in take_track_to_score:
                guess = deepcopy(take_track_to_score[f'{take_idx}_{track_idx}'])
            else:
                guess = {'Patient': 0.0003, 'head_surgeon': 0.0001, 'assistant_surgeon': 0.0002, 'circulating_nurse': 0.0005, 'anaesthetist': 0.0004}
            track_indices_to_guesses[track_idx] = guess

    # start with the most confident prediction first, assign it, continue (Should we choose highest score, or the most difference between first and second place), for now biggest difference
    while len(track_indices_to_guesses) > 0:
        highest_score = -1.
        highest_score_track_idx = None
        highest_score_guess = None
        for track_idx, guesses in track_indices_to_guesses.items():
            guess, score = max(guesses.items(), key=lambda x: x[1])
            if score > highest_score:
                highest_score_track_idx = track_idx
                highest_score_guess = guess
                highest_score = score

        human_name, joints = track_indices_to_human[highest_score_track_idx]
        assert human_name not in sg_human_name_to_roles
        sg_human_name_to_roles[human_name] = highest_score_guess
        sg_human_name_to_joints[human_name] = joints
        for _, guesses in track_indices_to_guesses.items():
            guesses[highest_score_guess] = 0.0

        del track_indices_to_guesses[highest_score_track_idx]

    return sg_human_name_to_roles, sg_human_name_to_joints


def name_to_index(name):
    name_to_index = {
        'Patient': 0,
        'head_surgeon': 1,
        'assistant_surgeon': 2,
        'circulating_nurse': 3,
        'anaesthetist': 4,
        'none': 5
    }
    return name_to_index[name]


def eval_role_prediction_perf(dataset, graphformer, dataset_name='role_prediction'):
    print('Starting SG Based Evaluation')
    # Score Order:
    take_track_to_score = {}
    take_to_results = {}
    takes_to_use = set()
    LABEL_NAMES = ['Patient', 'head_surgeon', 'assistant_surgeon', 'circulating_nurse', 'anaesthetist']

    loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=12,
                        pin_memory=True,
                        collate_fn=partial(collator, max_node=get_dataset(dataset_name)[
                            'max_node'], multi_hop_max_dist=5, spatial_pos_max=64, dataset_name=dataset_name),
                        )
    graphformer.eval()
    with torch.no_grad():
        for elem in loader:
            soft_score = F.softmax(graphformer(elem.to('cuda')) / 4, dim=1).cpu().numpy()[0]
            score_dict = {}
            for idx, role_name in enumerate(['Patient', 'head_surgeon', 'assistant_surgeon', 'circulating_nurse', 'anaesthetist']):
                score_dict[role_name] = soft_score[idx]
            take_track_to_score[f"{elem.meta['take_idx']}_{elem.meta['track_idx']}"] = score_dict
            takes_to_use.add(elem.meta['take_idx'])

    all_all_gt_labels = []
    all_all_pred_labels = []
    for take_idx in sorted(takes_to_use):
        all_gt_labels = []
        all_pred_labels = []
        root_path = OR_4D_DATA_ROOT_PATH / 'human_name_to_3D_joints'
        GT_take_human_name_to_3D_joints = np.load(str(root_path / f'{take_idx}_GT_True.npz'), allow_pickle=True)['arr_0'].item()
        with open(f'datasets/4D-OR/human_name_to_3D_joints/{take_idx}_scene_graph_track_GT_{USE_GT}.pickle', 'rb') as f:
            take_tracks = pickle.load(f)

        take_rels = get_take_rels(get_rels_path(take_idx, USE_GT, USE_IMAGES), take_idx, USE_GT)
        if len(take_rels) == 0:
            continue
        take_rels = sorted(take_rels, key=lambda x: x['scan'])
        # For evaluation, gt is still needed ofc
        gt_take_rels = get_take_rels(get_rels_path(take_idx, True, False), take_idx, True)
        gt_take_rels = sorted(gt_take_rels, key=lambda x: x['scan'])
        for sg, gt_sg in zip(take_rels, gt_take_rels):
            sg_humans_to_roles, sg_humans_to_joints = infer_roles_in_sg(sg, take_tracks, take_track_to_score, take_idx)
            if 'Patient' in gt_sg['objects'].values():
                gt_sg['human_idx_to_name']['Patient'] = 'Patient'

            if not USE_GT:
                # matching
                sg_humans_to_roles = match_human_preds_to_gt(GT_take_human_name_to_3D_joints[gt_sg['scan']], sg_humans_to_roles, sg_humans_to_joints)

            for human_idx, role in gt_sg['human_idx_to_name'].items():
                all_gt_labels.append(name_to_index(role.replace('-', '_')))
                pred_role = sg_humans_to_roles.get(human_idx, 'none')
                all_pred_labels.append(name_to_index(pred_role))

        result = classification_report(all_gt_labels, all_pred_labels, labels=list(range(len(LABEL_NAMES))), target_names=LABEL_NAMES, output_dict=True)
        take_to_results[take_idx] = {'micro_f1': result['accuracy'] if 'accuracy' in result else result['micro avg']['f1-score'],
                                     'macro_f1': result['macro avg']['f1-score']}
        result = classification_report(all_gt_labels, all_pred_labels, labels=list(range(len(LABEL_NAMES))), target_names=LABEL_NAMES)
        print(f'TAKE {take_idx}')
        print(result)
        all_all_gt_labels.extend(all_gt_labels)
        all_all_pred_labels.extend(all_pred_labels)

    return take_to_results, classification_report(all_all_gt_labels, all_all_pred_labels, labels=list(range(len(LABEL_NAMES))), target_names=LABEL_NAMES)


def output_role_predictions(graphformer, train_dataset, val_dataset, test_dataset, save_name):
    print('Outputting Roles')
    take_track_to_score = {}
    takes_to_use = set()
    output_json = {}

    for dataset in [train_dataset, val_dataset, test_dataset]:
        loader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=12,
                            pin_memory=True,
                            collate_fn=partial(collator, max_node=get_dataset('role_prediction')[
                                'max_node'], multi_hop_max_dist=5, spatial_pos_max=64, dataset_name='role_prediction'),
                            )
        graphformer.eval()
        with torch.no_grad():
            for elem in loader:
                soft_score = F.softmax(graphformer(elem.to('cuda')) / 4, dim=1).cpu().numpy()[0]
                score_dict = {}
                for idx, role_name in enumerate(['Patient', 'head_surgeon', 'assistant_surgeon', 'circulating_nurse', 'anaesthetist']):
                    score_dict[role_name] = soft_score[idx]
                take_track_to_score[f"{elem.meta['take_idx']}_{elem.meta['track_idx']}"] = score_dict
                takes_to_use.add(elem.meta['take_idx'])

        for take_idx in sorted(takes_to_use):
            print(f'Processing Take: {take_idx}')
            with open(f'datasets/4D-OR/human_name_to_3D_joints/{take_idx}_scene_graph_track_GT_{USE_GT}.pickle', 'rb') as f:
                take_tracks = pickle.load(f)

            take_rels = get_take_rels(get_rels_path(take_idx, USE_GT, USE_IMAGES), take_idx, USE_GT)
            if len(take_rels) == 0:
                continue
            take_rels = sorted(take_rels, key=lambda x: x['scan'])
            # For evaluation, gt is still needed ofc
            for sg in take_rels:
                sg_humans_to_roles, sg_humans_to_joints = infer_roles_in_sg(sg, take_tracks, take_track_to_score, take_idx)
                output_json[f'{sg["take_idx"]}_{sg["scan"]}'] = sg_humans_to_roles

    with open(save_name, 'w') as f:
        json.dump(output_json, f)
