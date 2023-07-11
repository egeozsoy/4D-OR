import json
import os
import torch
from pathlib import Path

import numpy as np
import open3d as o3d

from helpers.configurations import INSTANCE_LABELS_PATH, OBJECT_LABEL_MAP, INSTANCE_LABELS_PRED_PATH, TAKE_SPLIT, OR_4D_DATA_ROOT_PATH
from scene_graph_prediction.data_processing import compute_weight_occurrences
from scene_graph_prediction.utils import util


def dataset_loading(root: str, pth_selection: str, split: str, class_choice: list = None, USE_GT=False, for_infer=False):
    pth_catfile = os.path.join(pth_selection, 'classes.txt')
    classNames = util.read_txt_to_list(pth_catfile)

    pth_relationship = os.path.join(pth_selection, 'relationships.txt')
    util.check_file_exist(pth_relationship)
    relationNames = util.read_relationships(pth_relationship)

    selected_takes = set()
    if split == 'train':
        selected_takes = selected_takes.union(TAKE_SPLIT['train'])
    elif split == 'val':
        selected_takes = selected_takes.union(TAKE_SPLIT['val'])
    elif split == 'test':
        selected_takes = selected_takes.union(TAKE_SPLIT['test'])
    else:
        raise RuntimeError('unknown split type:', split)

    selected_scans = []
    for take_idx in selected_takes:
        pcd_paths = sorted(list(Path(f'{OR_4D_DATA_ROOT_PATH}/export_holistic_take{take_idx}_processed/pcds').glob('*.pcd')))
        for pcd_path in pcd_paths:
            selected_scans.append(f'{take_idx}_{pcd_path.name.replace(".pcd", "")}')

    with open(os.path.join(root, 'relationships_train.json'), "r") as read_file:
        data1 = json.load(read_file)
        data1['scans'] = data1['scans']
    with open(os.path.join(root, 'relationships_validation.json'), "r") as read_file:
        data2 = json.load(read_file)
        data2['scans'] = data2['scans']
    with open(os.path.join(root, 'relationships_test_dummy.json'),
              "r") as read_file:  # TODO if you have access to relationships_test.json, you can use that instead
        data3 = json.load(read_file)
        data3['scans'] = data3['scans']
    data = dict()
    data['scans'] = data1['scans'] + data2['scans'] + data3['scans']
    # data['neighbors'] = {**data1['neighbors'], **data2['neighbors']}
    label_to_object_map = {v: k for k, v in OBJECT_LABEL_MAP.items()}
    if for_infer and not USE_GT:
        for scan in list(data['scans']):
            instance_labels = np.load(str(INSTANCE_LABELS_PRED_PATH / f'{scan["take_idx"]}_{scan["scan"]}.npz'))['arr_0']
            labels = np.unique(instance_labels)
            scan_objects = [label_to_object_map[label] for label in labels if label >= 0]
            scan_objects.append('instrument')
            # scan_objects.append('object')
            scan_objects = {idx + 1: elem for idx, elem in enumerate(sorted(scan_objects))}
            scan['objects'] = scan_objects
            scan['relationships'] = []
    return classNames, relationNames, data, selected_scans


def load_mesh(scan_id_no_split, scan_id, objs_json, USE_GT=False, for_infer=False, human_name_to_3D_joints=None):
    take_idx, pcd_idx = scan_id_no_split.split('_')
    pcd_path = Path(f'{OR_4D_DATA_ROOT_PATH}/export_holistic_take{take_idx}_processed/pcds/{pcd_idx}.pcd')
    result = dict()
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    if USE_GT:
        instance_labels = np.load(str(INSTANCE_LABELS_PATH / f'{scan_id_no_split}.npz'))['arr_0']
    else:
        instance_labels_gt = np.load(str(INSTANCE_LABELS_PATH / f'{scan_id_no_split}.npz'))['arr_0']
        instance_labels = np.load(str(INSTANCE_LABELS_PRED_PATH / f'{scan_id_no_split}.npz'))['arr_0']
        if not for_infer:  # If for training or validation, match labels for humans + patient with GT, else doesn't make sense
            instance_labels = match_human_labels_to_gt(pcd, instance_labels_gt=instance_labels_gt, instance_labels_pred=instance_labels)
        del instance_labels_gt

    modified_instance_labels = np.zeros_like(instance_labels) - 1
    not_found_objects = {}
    instance_label_to_hand_locations = {}
    for key, value in sorted(objs_json[scan_id].items()):
        instance_label_idx = OBJECT_LABEL_MAP[value]
        # if np.sum(instance_labels == instance_label_idx) == 0 and value not in ['instrument', 'object']:
        if np.sum(instance_labels == instance_label_idx) == 0 and value not in ['instrument']:
            not_found_objects[key] = value
        modified_instance_labels[instance_labels == instance_label_idx] = key
        if human_name_to_3D_joints is not None and value in human_name_to_3D_joints[pcd_idx]:
            instance_label_to_hand_locations[key] = human_name_to_3D_joints[pcd_idx][value][8:10]

    result['points'] = np.concatenate([np.asarray(pcd.points), np.asarray(pcd.colors)], axis=1)
    result['instances'] = modified_instance_labels
    result['instance_label_to_hand_locations'] = instance_label_to_hand_locations

    # Adding virtual objects to the point cloud if necessary
    mesh_instrument = o3d.geometry.TriangleMesh.create_box(width=50.0, height=50.0, depth=50.0)
    mesh_instrument.paint_uniform_color([0.9, 0.1, 0.1])
    mesh_instrument.translate([-200, 1056, -66])
    instrument_pc = mesh_instrument.sample_points_uniformly(1000)
    instrument_points = np.concatenate([np.asarray(instrument_pc.points), np.asarray(instrument_pc.colors)], axis=1)
    instrument_labels = np.zeros(len(instrument_points), dtype=np.int8) + list(objs_json[scan_id].keys())[list(objs_json[scan_id].values()).index('instrument')]

    result['points'] = np.concatenate([result['points'], instrument_points], axis=0)
    result['instances'] = np.concatenate([result['instances'], instrument_labels], axis=0)

    for key, value in not_found_objects.items():
        print(f"{value} not in the scan, emulating")
        tmp = o3d.geometry.TriangleMesh.create_cone(radius=50)
        tmp.paint_uniform_color([0.9, 0.9, 0.1])
        tmp.translate([600, 1056, -66])
        pc_tmp = tmp.sample_points_uniformly(1000)
        points_tmp = np.concatenate([np.asarray(pc_tmp.points), np.asarray(pc_tmp.colors)], axis=1)
        labels_tmp = np.zeros(len(points_tmp), dtype=np.int8) + key
        result['points'] = np.concatenate([result['points'], points_tmp], axis=0)
        result['instances'] = np.concatenate([result['instances'], labels_tmp], axis=0)

    return result


def compute_dist_matrix_between_human_pcds(gt_humans, pred_humans, DOWNSAMPLE_COUNT=1000):
    dist_matrix = np.zeros((len(gt_humans), len(pred_humans)))
    for idx, gt_human in enumerate(gt_humans):
        for idy, pred_human in enumerate(pred_humans):
            # downsample for speed reasons
            choices = np.random.choice(len(gt_human[1]), DOWNSAMPLE_COUNT, replace=len(gt_human[1]) < DOWNSAMPLE_COUNT)
            gt_human_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_human[1][choices]))
            choices = np.random.choice(len(pred_human[1]), DOWNSAMPLE_COUNT, replace=len(pred_human[1]) < DOWNSAMPLE_COUNT)
            pred_human_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred_human[1][choices]))
            dist = np.asarray(gt_human_pcd.compute_point_cloud_distance(pred_human_pcd)).mean()
            dist_matrix[idx, idy] = dist
    return dist_matrix


def match_human_labels_to_gt(pcd, instance_labels_gt, instance_labels_pred):
    gt_pred_label_matches = []
    gt_humans = []
    HUMAN_NAMES = ['Patient', 'human_0', 'human_1', 'human_2', 'human_3', 'human_4', 'human_5', 'human_6']
    for human_name in HUMAN_NAMES:
        instance_label_idx = OBJECT_LABEL_MAP[human_name]
        if np.sum(instance_labels_gt == instance_label_idx) > 0:
            gt_humans.append((instance_label_idx, np.asarray(pcd.points)[instance_labels_gt == instance_label_idx]))
    pred_humans = []
    for human_name in HUMAN_NAMES:
        instance_label_idx = OBJECT_LABEL_MAP[human_name]
        if np.sum(instance_labels_pred == instance_label_idx) > 0:
            pred_humans.append((instance_label_idx, np.asarray(pcd.points)[instance_labels_pred == instance_label_idx]))

    while len(gt_humans) > 0 and len(pred_humans) > 0:
        dist_matrix = compute_dist_matrix_between_human_pcds(gt_humans=gt_humans, pred_humans=pred_humans)
        min_value_index = np.unravel_index(dist_matrix.argmin(), dist_matrix.shape)
        gt_human_idx = min_value_index[0]
        pred_human_idx = min_value_index[1]
        gt_pred_label_matches.append((gt_humans[gt_human_idx][0], pred_humans[pred_human_idx][0]))
        gt_humans.pop(gt_human_idx)
        pred_humans.pop(pred_human_idx)

    instance_labels_pred_modified = instance_labels_pred.copy()

    # First delete all the existing instance labels
    for human_name in HUMAN_NAMES:
        instance_label_idx = OBJECT_LABEL_MAP[human_name]
        instance_labels_pred_modified[instance_labels_pred_modified == instance_label_idx] = -1

    # Add new instance labels for humans
    for gt_pred_label_match in gt_pred_label_matches:
        instance_label_gt, instance_label_pred = gt_pred_label_match
        instance_labels_pred_modified[instance_labels_pred == instance_label_pred] = instance_label_gt

    used_gt_instance_labels = {elem[0] for elem in gt_pred_label_matches}
    # add remaining pred_humans if there are still non matching ones
    if len(pred_humans) > 0:
        print('Unmatched human pred found')
        for non_matched_human in pred_humans:
            instance_label_idx = non_matched_human[0]
            # Decide on instance label
            for human_name in HUMAN_NAMES:
                new_instance_label_idx = OBJECT_LABEL_MAP[human_name]
                if new_instance_label_idx not in used_gt_instance_labels:
                    instance_labels_pred_modified[instance_labels_pred == instance_label_idx] = new_instance_label_idx

    return instance_labels_pred_modified


def load_data(root, config, mconfig, split, for_eval):
    classNames = None
    relationNames = None
    data = None
    selected_scans = None
    for i in range(len(root)):
        selection = root[i]
        l_classNames, l_relationNames, l_data, l_selected_scans = \
            dataset_loading(root[i], selection, split, USE_GT=config['USE_GT'], for_infer=for_eval)

        if classNames is None:
            classNames, relationNames, data, selected_scans = \
                l_classNames, l_relationNames, l_data, l_selected_scans
        else:
            classNames = set(classNames).union(l_classNames)
            relationNames = set(relationNames).union(l_relationNames)
            data['scans'] = l_data['scans'] + data['scans']
            data['neighbors'] = {**l_data['neighbors'], **data['neighbors']}
            selected_scans = selected_scans.union(l_selected_scans)
    classNames = list(classNames)
    relationNames = list(relationNames)

    relationNames = sorted(relationNames)
    classNames = sorted(classNames)

    if 'none' not in relationNames:
        relationNames.append('none')

    return classNames, relationNames, data, selected_scans


def get_relationships(data, selected_scans: list, classNames, verbose=True):
    rel = dict()
    objs = dict()
    scans = list()
    for scan in data['scans']:
        if scan['take_idx'] in TAKE_SPLIT['train']:
            split = 0
        elif scan['take_idx'] in TAKE_SPLIT['val']:
            split = 1
        elif scan['take_idx'] in TAKE_SPLIT['test']:
            split = 2

        if f'{scan["take_idx"]}_{scan["scan"]}' not in selected_scans:
            continue

        relationships = []
        for realationship in scan["relationships"]:
            relationships.append(realationship)

        objects = {}
        for k, v in scan["objects"].items():
            objects[int(k)] = v

        # filter scans that doesn't have the classes we care
        instances_id = list(objects.keys())
        valid_counter = 0
        for instance_id in instances_id:
            instance_labelName = objects[instance_id]
            if instance_labelName in classNames:  # is it a class we care about?
                valid_counter += 1
        if valid_counter < 3:  # need at least three nodes
            continue

        rel[f'{scan["take_idx"]}_{scan["scan"]}' + "_" + str(split)] = relationships
        scans.append(f'{scan["take_idx"]}_{scan["scan"]}' + "_" + str(split))

        objs[f'{scan["take_idx"]}_{scan["scan"]}' + "_" + str(split)] = objects

    if verbose:
        print('num of data:', len(scans))

    return rel, objs, scans


def get_weights(classNames, relationNames, data, selected_scans, verbose=True, for_eval=False):
    if for_eval:
        return None, None
    wobjs, wrels, o_obj_cls, o_rel_cls = compute_weight_occurrences.compute(classNames, relationNames, data, selected_scans, False)
    w_cls_obj = torch.from_numpy(np.array(o_obj_cls)).float()
    w_cls_rel = torch.from_numpy(np.array(o_rel_cls)).float()
    w_cls_obj = torch.abs(1.0 / (torch.log(w_cls_obj) + 1))  # +1 to prevent 1 /log(1) = inf
    # w_cls_rel = torch.abs(1.0 / (torch.log(w_cls_rel) + 1))  # +1 to prevent 1 /log(1) = inf, log weighting
    w_cls_rel = 1.0 / w_cls_rel  # linear weighting

    w_cls_rel[-1] = 0.0001  # weight of 'none' relation

    if verbose:
        print('=== {} classes ==='.format(len(classNames)))
        for i in range(len(classNames)):
            print('|{0:>2d} {1:>20s}'.format(i, classNames[i]), end='')
            if w_cls_obj is not None:
                print(':{0:>1.3f}|'.format(w_cls_obj[i]), end='')
            if (i + 1) % 2 == 0:
                print('')
        print('')
        print('=== {} relationships ==='.format(len(relationNames)))
        for i in range(len(relationNames)):
            print('|{0:>2d} {1:>20s}'.format(i, relationNames[i]), end=' ')
            if w_cls_rel is not None:
                print('{0:>1.3f}|'.format(w_cls_rel[i]), end='')
            if (i + 1) % 2 == 0:
                print('')
        print('')

    return w_cls_obj, w_cls_rel
