import json
from pathlib import Path
from typing import List, Dict

from sklearn.metrics import classification_report

from helpers.configurations import TAKE_SPLIT, OR_4D_DATA_ROOT_PATH
import pickle

from collections import defaultdict
from copy import deepcopy
import numpy as np


def convert_scene_graph_to_human_readable(scan_gt_rels):
    object_idx_to_name = scan_gt_rels['objects']
    human_readable_relationships = []
    for sub_idx, obj_idx, rel_idx, rel_name in scan_gt_rels['relationships']:
        sub_name = object_idx_to_name[str(sub_idx)]
        obj_name = object_idx_to_name[str(obj_idx)]
        if 'human' in sub_name or 'Patient' in sub_name:
            sub_name = 'human'
        if 'human' in obj_name or 'Patient' in obj_name:
            obj_name = 'human'
        human_readable_relationships.append((sub_name, rel_name, obj_name))

    return human_readable_relationships


def rel_counter_in_relationships(relationships, sub=None, rel=None, obj=None):
    count = 0
    for s, r, o in relationships:
        if sub is not None and sub != s:
            continue
        if rel is not None and rel != r:
            continue
        if obj is not None and obj != o:
            continue
        count += 1
    return count


def check_getting_applied_patient_actions(rels, role_guesses):
    total_count = 0.
    total_count += rel_counter_in_relationships(rels, rel='Cementing', obj='TARGET')
    total_count += rel_counter_in_relationships(rels, rel='Cutting', obj='TARGET')
    total_count += rel_counter_in_relationships(rels, rel='Drilling', obj='TARGET')
    total_count += rel_counter_in_relationships(rels, rel='Hammering', obj='TARGET')
    total_count += rel_counter_in_relationships(rels, rel='Sawing', obj='TARGET')
    total_count += rel_counter_in_relationships(rels, rel='Suturing', obj='TARGET')
    total_count += rel_counter_in_relationships(rels, rel='Cleaning', obj='TARGET')
    total_count += rel_counter_in_relationships(rels, rel='Preparing', obj='TARGET')
    role_guesses['Patient'] += total_count * 10  # times 10 because this is a very clear signal that it is the patient


def check_applying_head_surgeon_actions(rels, role_guesses):
    total_count = 0.
    total_count += rel_counter_in_relationships(rels, rel='Cementing', sub='TARGET')
    total_count += rel_counter_in_relationships(rels, rel='Cutting', sub='TARGET')
    total_count += rel_counter_in_relationships(rels, rel='Drilling', sub='TARGET')
    total_count += rel_counter_in_relationships(rels, rel='Hammering', sub='TARGET')
    total_count += rel_counter_in_relationships(rels, rel='Sawing', sub='TARGET')
    total_count += rel_counter_in_relationships(rels, rel='Suturing', sub='TARGET')
    role_guesses['head_surgeon'] += total_count * 10


def check_rels(rels, role_guesses):
    # Check clear indications of patient
    lying_on_count = rel_counter_in_relationships(rels, sub='TARGET', rel='LyingOn', obj='operating_table')
    role_guesses['Patient'] += lying_on_count * 10  # times 10 because this is a very clear signal that it is the patient
    check_getting_applied_patient_actions(rels, role_guesses)
    # Check clear indications of head surgeon
    check_applying_head_surgeon_actions(rels, role_guesses)
    # Check clear indication of anaesthetist
    anesthesia_operating_count = 0.
    anesthesia_operating_count += rel_counter_in_relationships(rels, sub='TARGET', rel='Operating', obj='anesthesia_equipment')
    anesthesia_operating_count += rel_counter_in_relationships(rels, sub='TARGET', rel='Touching', obj='anesthesia_equipment')
    role_guesses['anaesthetist'] += anesthesia_operating_count * 50

    # Check likely indication of head surgeon
    touching_count = rel_counter_in_relationships(rels, sub='TARGET', rel='Touching', obj='human')
    role_guesses['head_surgeon'] += touching_count * 5
    role_guesses['assistant_surgeon'] += touching_count

    # Check likely indication of assistant
    assisting_count = rel_counter_in_relationships(rels, sub='TARGET', rel='Assisting', obj='human')
    role_guesses['assistant_surgeon'] += assisting_count * 10
    role_guesses['circulating_nurse'] += assisting_count

    # Check likely indication of getting assisted
    getting_assisted_count = rel_counter_in_relationships(rels, sub='human', rel='Assisting', obj='TARGET') / 2
    role_guesses['head_surgeon'] += getting_assisted_count * 4
    role_guesses['assistant_surgeon'] += getting_assisted_count

    # Check indication of assistant or head surgeon
    cleaning_count = rel_counter_in_relationships(rels, sub='TARGET', rel='Cleaning', obj='human')
    role_guesses['assistant_surgeon'] += cleaning_count * 10
    role_guesses['head_surgeon'] += cleaning_count * 5

    # Check preparing
    preparing_count = rel_counter_in_relationships(rels, sub='TARGET', rel='Preparing', obj='human')
    role_guesses['assistant_surgeon'] += preparing_count * 5
    role_guesses['head_surgeon'] += preparing_count * 5
    role_guesses['circulating_nurse'] += preparing_count * 2
    role_guesses['anaesthetist'] += preparing_count

    # Check indication of anaesthetist or circulating_nurse
    operating_op_table_count = rel_counter_in_relationships(rels, sub='TARGET', rel='Operating', obj='operating_table')
    role_guesses['anaesthetist'] += operating_op_table_count * 10
    role_guesses['circulating_nurse'] += operating_op_table_count * 10

    # Likely indication of circulating nurse or assistant surgeon
    interaction_instrument_table_count = 0.
    interaction_instrument_table_count += rel_counter_in_relationships(rels, sub='TARGET', rel='Touching', obj='instrument_table')
    interaction_instrument_table_count += rel_counter_in_relationships(rels, sub='TARGET', rel='CloseTo', obj='instrument_table')
    role_guesses['assistant_surgeon'] += interaction_instrument_table_count * 3
    role_guesses['circulating_nurse'] += interaction_instrument_table_count * 1

    # Likely indication of circulating nurse or anaesthetist
    interaction_secondary_table_count = 0.
    interaction_secondary_table_count += rel_counter_in_relationships(rels, sub='TARGET', rel='Touching', obj='secondary_table')
    interaction_secondary_table_count += rel_counter_in_relationships(rels, sub='TARGET', rel='CloseTo', obj='secondary_table')
    role_guesses['circulating_nurse'] += interaction_secondary_table_count * 8
    role_guesses['anaesthetist'] += interaction_secondary_table_count * 1

    # Likely indication holding instrument
    holding_count = 0.
    holding_count += rel_counter_in_relationships(rels, sub='TARGET', rel='Holding', obj='instrument')
    role_guesses['head_surgeon'] += holding_count * 5
    role_guesses['assistant_surgeon'] += holding_count * 4
    role_guesses['circulating_nurse'] += holding_count * 1

    # If close to anaesthetie_machine, likely patinet or anaesthetist, use little weighting so patient with lyingon etc. relations will come at the front
    anest_close_count = rel_counter_in_relationships(rels, sub='TARGET', rel='CloseTo', obj='anesthesia_equipment')
    role_guesses['anaesthetist'] += anest_close_count

    # Weak indication towards head_surgeon or assistant
    op_table_close_count = rel_counter_in_relationships(rels, sub='TARGET', rel='CloseTo', obj='operating_table')
    role_guesses['head_surgeon'] += op_table_close_count
    role_guesses['assistant_surgeon'] += op_table_close_count

    # Add default values for ordering if no other information is available
    role_guesses['circulating_nurse'] += 0.005
    role_guesses['anaesthetist'] += 0.004
    role_guesses['Patient'] += 0.003
    role_guesses['assistant_surgeon'] += 0.002
    role_guesses['head_surgeon'] += 0.001


def get_rels_path(take_idx, USE_GT, USE_IMAGES):
    if take_idx in TAKE_SPLIT['train']:
        if USE_GT:
            return Path('data/relationships_train.json')
        else:
            return Path('scan_relations_training_no_gt_train_scans.json') if not USE_IMAGES else Path('scan_relations_training_no_gt_images_train_scans.json')
    elif take_idx in TAKE_SPLIT['val']:
        if USE_GT:
            return Path('data/relationships_validation.json')
        else:
            return Path('scan_relations_training_no_gt_validation_scans.json') if not USE_IMAGES else Path(
                'scan_relations_training_no_gt_images_validation_scans.json')
    elif take_idx in TAKE_SPLIT['test']:
        if USE_GT:
            return Path('data/relationships_test_dummy.json')
        else:
            return Path('scan_relations_training_no_gt_test_scans.json') if not USE_IMAGES else Path('scan_relations_training_no_gt_images_test_scans.json')

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


def get_track_rels(take_tracks: List[Dict[str, int]], take_rels: List[Dict[str, int]], USE_GT: bool):
    track_rel = {}
    for track_idx, track in enumerate(take_tracks):
        # Collect all scene graphs matching to this target, rename target node to human_target
        all_corresponding_scene_graphs = []
        for frame_str, (h_name, h_joint) in track['timestamp_to_human_pose'].items():
            matching_scene_graph = deepcopy([take_rel for take_rel in take_rels if take_rel['scan'] == frame_str][0])
            if USE_GT:
                matching_scene_graph['objects'] = {k: v.replace(h_name, 'TARGET') for k, v in matching_scene_graph['objects'].items()}
            else:
                for rel in matching_scene_graph['relationships']:
                    if rel[0] == h_name:
                        rel[0] = 'TARGET'
                    elif rel[2] == h_name:
                        rel[2] = 'TARGET'
            all_corresponding_scene_graphs.append(matching_scene_graph)

        rels = []
        for sg in all_corresponding_scene_graphs:
            relevant_rels = []
            if USE_GT:
                human_readable_relationships = convert_scene_graph_to_human_readable(sg)
            else:
                human_readable_relationships = sg['relationships']
            for rel in human_readable_relationships:
                if rel[0] == 'TARGET' or rel[2] == 'TARGET':
                    relevant_rels.append(rel)

            rels.extend(relevant_rels)
        track_rel[track_idx] = rels
    return track_rel


def calculate_guesses_for_tracks(track_rel):
    track_to_guesses = {}
    for track_idx, rels, in track_rel.items():
        role_guesses = defaultdict(float)
        check_rels(rels, role_guesses)
        # Normalize if necessary
        total = sum(role_guesses.values())
        if total > 1.0:
            role_guesses = {k: v / total for k, v in role_guesses.items()}
        track_to_guesses[track_idx] = role_guesses  # We are not argmaxing here, because this information might come handy
    return track_to_guesses


def infer_roles_in_sg(sg, take_tracks, track_to_guesses):
    frame_str = sg['scan']
    track_indices_to_human = {}
    track_indices_to_guesses = {}
    sg_human_name_to_roles = {}
    sg_human_name_to_joints = {}
    for track_idx, track in enumerate(take_tracks):
        if frame_str in track['timestamp_to_human_pose']:
            track_indices_to_human[track_idx] = deepcopy(track['timestamp_to_human_pose'][frame_str])
            track_indices_to_guesses[track_idx] = deepcopy(track_to_guesses[track_idx])

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


def main():
    USE_GT_SCENE_GRAPHS = False
    USE_IMAGES = True
    print(f'USE GT : {USE_GT_SCENE_GRAPHS}')
    print(f'USE IMAGES : {USE_IMAGES}')
    LABEL_NAMES = ['Patient', 'head_surgeon', 'assistant_surgeon', 'circulating_nurse', 'anaesthetist']
    output_json = {}
    split_to_all_gt_labels = defaultdict(list)
    split_to_all_pred_labels = defaultdict(list)

    for take_idx in TAKE_SPLIT['train'] + TAKE_SPLIT['val'] + TAKE_SPLIT['test']:
        root_path = OR_4D_DATA_ROOT_PATH / 'human_name_to_3D_joints'
        try:
            GT_take_human_name_to_3D_joints = np.load(str(root_path / f'{take_idx}_GT_True.npz'), allow_pickle=True)['arr_0'].item()
        except FileNotFoundError:
            print(f'{root_path / f"{take_idx}_GT_True.npz"} not found')
            continue
        all_gt_labels = []
        all_pred_labels = []
        with open(f'datasets/4D-OR/human_name_to_3D_joints/{take_idx}_scene_graph_track_GT_{USE_GT_SCENE_GRAPHS}.pickle', 'rb') as f:
            take_tracks = pickle.load(f)
        rels_path = get_rels_path(take_idx, USE_GT_SCENE_GRAPHS, USE_IMAGES)

        take_rels = get_take_rels(rels_path, take_idx, USE_GT_SCENE_GRAPHS)
        if take_rels is None:
            continue

        take_rels = sorted(take_rels, key=lambda x: x['scan'])
        if len(take_rels) == 0:
            continue

        track_rel = get_track_rels(take_tracks, take_rels, USE_GT_SCENE_GRAPHS)
        track_to_guesses = calculate_guesses_for_tracks(track_rel)

        # For evaluation, gt is still needed ofc
        gt_take_rels = get_take_rels(get_rels_path(take_idx, True, USE_IMAGES), take_idx, True)
        for sg, gt_sg in zip(take_rels, gt_take_rels):
            sg_humans_to_roles, sg_humans_to_joints = infer_roles_in_sg(sg, take_tracks, track_to_guesses)
            output_json[f'{sg["take_idx"]}_{sg["scan"]}'] = sg_humans_to_roles
            if 'Patient' in gt_sg['objects'].values():
                gt_sg['human_idx_to_name']['Patient'] = 'Patient'

            if not USE_GT_SCENE_GRAPHS:
                # matching
                sg_humans_to_roles = match_human_preds_to_gt(GT_take_human_name_to_3D_joints[gt_sg['scan']], sg_humans_to_roles, sg_humans_to_joints)

            for human_idx, role in gt_sg['human_idx_to_name'].items():
                all_gt_labels.append(name_to_index(role.replace('-', '_')))
                pred_role = sg_humans_to_roles.get(human_idx, 'none')
                all_pred_labels.append(name_to_index(pred_role))

        result = classification_report(all_gt_labels, all_pred_labels, labels=list(range(len(LABEL_NAMES))), target_names=LABEL_NAMES)
        print(f'TAKE {take_idx}')
        print(result)
        if take_idx in TAKE_SPLIT['train']:
            split_to_all_gt_labels['train'].extend(all_gt_labels)
            split_to_all_pred_labels['train'].extend(all_pred_labels)
        elif take_idx in TAKE_SPLIT['val']:
            split_to_all_gt_labels['val'].extend(all_gt_labels)
            split_to_all_pred_labels['val'].extend(all_pred_labels)
        else:
            split_to_all_gt_labels['test'].extend(all_gt_labels)
            split_to_all_pred_labels['test'].extend(all_pred_labels)

    train_results = classification_report(split_to_all_gt_labels['train'], split_to_all_pred_labels['train'], labels=list(range(len(LABEL_NAMES))),
                                          target_names=LABEL_NAMES)
    val_results = classification_report(split_to_all_gt_labels['val'], split_to_all_pred_labels['val'], labels=list(range(len(LABEL_NAMES))),
                                        target_names=LABEL_NAMES)
    test_results = classification_report(split_to_all_gt_labels['test'], split_to_all_pred_labels['test'], labels=list(range(len(LABEL_NAMES))),
                                         target_names=LABEL_NAMES)
    print(f'TRAIN')
    print(train_results)
    print(f'VAL')
    print(val_results)
    print(f'TEST')
    print(test_results)

    with open(f'rule_based_role_predictions_with_GT_{USE_GT_SCENE_GRAPHS}_{USE_IMAGES}.json', 'w') as f:
        json.dump(output_json, f)


if __name__ == '__main__':
    main()
