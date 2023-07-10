import json
from pathlib import Path

import numpy as np
import torch, random
import open3d as o3d
from PIL import Image

from helpers.configurations import OR_4D_DATA_ROOT_PATH


def zero_mean(point):
    mean = torch.mean(point, dim=0)
    point -= mean.unsqueeze(0)
    # Norm to unit sphere
    furthest_distance = point.pow(2).sum(1).sqrt().max()  # find maximum distance for each n -> [n]
    point /= furthest_distance
    return point, {'mean': mean.unsqueeze(0), 'dist': furthest_distance}


def objname_to_onehot(objname):
    obj_name_to_index = {
        'anesthesia_equipment': 0,
        'operating_table': 1,
        'instrument_table': 2,
        'secondary_table': 3,
        'instrument': 4,
        'human': 5
    }
    if 'human' in objname or 'Patient' in objname:  # We don't care about patient human_0 human_1 etc. everything is human (We don't seperate patient here, because voxelpose also won't seperate it)
        objname = 'human'
    one_hot_vector = np.zeros(len(obj_name_to_index))
    one_hot_vector[obj_name_to_index[objname]] = 1
    return one_hot_vector


def calculate_downsample_indices(pointset, target_N):
    if len(pointset) < target_N:
        return np.random.choice(len(pointset), target_N, replace=True)

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pointset[:, :3]))
    best_choice = np.asarray(list(range(len(pointset))))
    for sampling_size in range(15, 100, 5):
        choice = np.unique(pc.voxel_down_sample_and_trace(sampling_size, pc.get_min_bound(), pc.get_max_bound())[1])[1:]
        if len(choice) > target_N:
            best_choice = choice
        else:
            break
    return best_choice[np.random.choice(len(best_choice), target_N, replace=False)]


def data_preparation(config, points, instances, selected_instances, num_points, num_points_union,
                     # use_rgb, use_normal,
                     for_train=False, instance2labelName=None, classNames=None,
                     rel_json=None, relationships=None, multi_rel_outputs=False,
                     padding=0.2, shuffle_objs=True, instance_label_to_hand_locations=None, cam_infos=None):
    if for_train:
        assert (instance2labelName is not None)
        assert (rel_json is not None)
        assert (classNames is not None)
        assert (relationships is not None)
        assert (multi_rel_outputs is not None)

    instances = instances.flatten()

    instances_id = list(np.unique(instances))

    if 0 in instances_id:
        instances_id.remove(0)
    if shuffle_objs:
        random.shuffle(instances_id)

    instance2mask = {}
    instance2mask[0] = 0
    cat = []
    counter = 0
    ''' Build instance2mask and their gt classes '''
    for instance_id in list(np.unique(instances)):
        # print('instance {} size: {}'.format(instance_id,len(points[np.where(instances == instance_id)])))
        if selected_instances is not None:
            if instance_id not in selected_instances:
                # since we divide the whole graph into sub-graphs if the
                # scene graph is too large to resolve the memory issue, there
                # are instances that are not interested in the current sub-graph
                instance2mask[instance_id] = 0
                continue

        if for_train:
            class_id = -1  # was 0

            instance_labelName = instance2labelName[instance_id]
            if instance_labelName in classNames:  # is it a class we care about?
                class_id = classNames.index(instance_labelName)

            if (class_id >= 0) and (instance_id > 0):  # there is no instance 0?
                cat.append(class_id)
        else:
            class_id = 0

        if class_id != -1:  # was 0
            counter += 1
            instance2mask[instance_id] = counter
        else:
            instance2mask[instance_id] = 0

    num_objects = len(instances_id) if selected_instances is None else len(selected_instances)
    masks = np.asarray([instance2mask[instance] for instance in instances], dtype=np.int32)
    mask2_instance = {v: k for k, v in instance2mask.items()}
    dim_point = points.shape[1]
    obj_points = torch.zeros([num_objects, num_points, dim_point])
    obj_bboxes = list()
    obj_masks = list()
    obj_cam_limits = []

    for i in range(num_objects):
        obj_pointset = points[np.where(masks == i + 1)[0], :]
        min_box = np.min(obj_pointset[:, :3], 0) - padding
        max_box = np.max(obj_pointset[:, :3], 0) + padding
        obj_bboxes.append([min_box, max_box])
        # choice = np.random.choice(len(obj_pointset), num_points, replace=True)
        choice = calculate_downsample_indices(obj_pointset, num_points)
        obj_pointset = obj_pointset[choice, :]
        obj_pointset = torch.from_numpy(obj_pointset.astype(np.float32))
        obj_pointset[:, :3], _ = zero_mean(obj_pointset[:, :3])
        obj_points[i] = obj_pointset

    # Build fully connected edges
    edge_indices = list()
    max_edges = -1
    for n in range(len(cat)):
        for m in range(len(cat)):
            if n == m: continue
            edge_indices.append([n, m])
    if max_edges > 0 and len(edge_indices) > max_edges and for_train:
        # for eval, do not drop out any edges.
        indices = list(np.random.choice(len(edge_indices), max_edges))
        edge_indices = edge_indices[indices]

    if for_train:
        ''' Build rel class GT '''
        if multi_rel_outputs:
            adj_matrix_onehot = np.zeros([num_objects, num_objects, len(relationships)])
        else:
            adj_matrix = np.zeros([num_objects, num_objects]) + relationships.index('none')  # Default should be none

        for r in rel_json:
            if r[0] not in instance2mask or r[1] not in instance2mask: continue
            index1 = instance2mask[r[0]] - 1
            index2 = instance2mask[r[1]] - 1
            if for_train:
                if r[3] not in relationships:
                    continue
                r[2] = relationships.index(r[3])  # remap the index of relationships in case of custom relationNames

            if index1 >= 0 and index2 >= 0:
                if multi_rel_outputs:
                    adj_matrix_onehot[index1, index2, r[2]] = 1
                else:
                    adj_matrix[index1, index2] = r[2]

        ''' Build rel point cloud '''
        if multi_rel_outputs:
            rel_dtype = np.float32
            adj_matrix_onehot = torch.from_numpy(np.array(adj_matrix_onehot, dtype=rel_dtype))
        else:
            rel_dtype = np.int64

        if multi_rel_outputs:
            gt_rels = torch.zeros(len(edge_indices), len(relationships), dtype=torch.float)
        else:
            gt_rels = torch.zeros(len(edge_indices), dtype=torch.long)

    rel_points = list()
    rel_masks = list()
    rel_cam_limits = []
    rel_hand_points = torch.zeros(len(edge_indices), 2, 3, dtype=torch.float16)
    relation_objects_one_hot = []
    for e in range(len(edge_indices)):
        edge = edge_indices[e]
        index1 = edge[0]
        index2 = edge[1]
        obj1_instance = mask2_instance[index1 + 1]
        obj2_instance = mask2_instance[index2 + 1]
        obj1_name = instance2labelName[obj1_instance]
        obj2_name = instance2labelName[obj2_instance]
        if for_train:
            if multi_rel_outputs:
                gt_rels[e, :] = adj_matrix_onehot[index1, index2, :]
            else:
                gt_rels[e] = adj_matrix[index1, index2]
                # rel_name = relationships[gt_rels[e].item()]
                # if rel_name != 'none':
                #     print(f'{obj1_name} -{rel_name}> {obj2_name}')

        obj1_one_hot = objname_to_onehot(obj1_name)
        obj2_one_hot = objname_to_onehot(obj2_name)
        relation_objects_one_hot.append(np.concatenate([obj1_one_hot, obj2_one_hot]))

        mask1 = (masks == index1 + 1).astype(np.int32) * 1
        mask2 = (masks == index2 + 1).astype(np.int32) * 2
        mask_ = np.expand_dims(mask1 + mask2, 1)
        bbox1 = obj_bboxes[index1]
        bbox2 = obj_bboxes[index2]
        min_box = np.minimum(bbox1[0], bbox2[0])
        max_box = np.maximum(bbox1[1], bbox2[1])
        filter_mask = (points[:, 0] > min_box[0]) * (points[:, 0] < max_box[0]) \
                      * (points[:, 1] > min_box[1]) * (points[:, 1] < max_box[1]) \
                      * (points[:, 2] > min_box[2]) * (points[:, 2] < max_box[2])

        points4d = np.concatenate([points, mask_], 1)
        pointset = points4d[np.where(filter_mask > 0)[0], :]
        # choice = np.random.choice(len(pointset), num_points_union, replace=True)
        choice = calculate_downsample_indices(pointset, num_points_union)
        pointset = pointset[choice, :]

        pointset = torch.from_numpy(pointset.astype(np.float32))
        pointset[:, :3], info = zero_mean(pointset[:, :3])
        rel_points.append(pointset)
        if instance_label_to_hand_locations is not None and obj1_instance in instance_label_to_hand_locations:
            hand_points = torch.from_numpy(instance_label_to_hand_locations[obj1_instance]).clone()
            hand_points -= info['mean']
            hand_points /= info['dist']
            rel_hand_points[e] = hand_points

    relation_objects_one_hot = torch.from_numpy(np.stack(relation_objects_one_hot)).float()
    cat = torch.from_numpy(np.array(cat, dtype=np.int64))
    edge_indices = torch.tensor(edge_indices, dtype=torch.long)

    if for_train:
        try:
            rel_points = torch.stack(rel_points, 0)
        except:
            rel_points = torch.zeros([4, num_points_union])
    else:
        rel_points = torch.stack(rel_points, 0)

    if for_train:
        return obj_points, rel_points, edge_indices, instance2mask, relation_objects_one_hot, gt_rels, cat, rel_hand_points
    else:
        return obj_points, rel_points, edge_indices, instance2mask, relation_objects_one_hot, rel_hand_points


def visualize_data(obj_points, gt_class, classNames, relationNames, rel_points, edge_indices, gt_rels, map_instance2labelName):
    for obj_point, gt_cls in zip(obj_points, gt_class):
        obj_name = classNames[gt_cls.item()]
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(obj_point.cpu().detach().numpy()[:, :3])
        pc.colors = o3d.utility.Vector3dVector(obj_point.cpu().detach().numpy()[:, 3:6])
        o3d.io.write_point_cloud(f'debugging_ply/{obj_name}_no_aug.ply', pc)

    for rel_point, edge_index, gt_rel in zip(rel_points, edge_indices, gt_rels):
        if gt_rel.item() == relationNames.index('none'):
            continue
        else:
            rel_name = relationNames[gt_rel.item()]

        rel_from = map_instance2labelName[edge_index[0].item() + 1]
        rel_to = map_instance2labelName[edge_index[1].item() + 1]
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(rel_point.cpu().detach().numpy()[:, :3])
        pc.colors = o3d.utility.Vector3dVector(rel_point.cpu().detach().numpy()[:, 3:6])
        o3d.io.write_point_cloud(f'debugging_ply/{rel_from}_{rel_to}_{rel_name}_no_aug.ply', pc)

    for edge, rel in zip(edge_indices, gt_rels):
        if rel == relationNames.index('none'):
            continue
        start = edge[0]
        end = edge[1]
        start_name = map_instance2labelName[start.item() + 1]
        end_name = map_instance2labelName[end.item() + 1]
        rel_name = relationNames[rel]
        print((start_name, rel_name, end_name))

    # Can be used for debugging / visualizing
    for obj_point, gt_cls in zip(obj_points, gt_class):
        obj_name = classNames[gt_cls.item()]
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(obj_point.cpu().detach().numpy()[:, :3])
        pc.colors = o3d.utility.Vector3dVector(obj_point.cpu().detach().numpy()[:, 3:6])
        o3d.io.write_point_cloud(f'debugging_ply/{obj_name}.ply', pc)

    for rel_point, edge_index, gt_rel in zip(rel_points, edge_indices, gt_rels):
        if gt_rel.item() == relationNames.index('none'):
            continue
        else:
            rel_name = relationNames[gt_rel.item()]

        rel_from = map_instance2labelName[edge_index[0].item() + 1]
        rel_to = map_instance2labelName[edge_index[1].item() + 1]
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(rel_point.cpu().detach().numpy()[:, :3])
        pc.colors = o3d.utility.Vector3dVector(rel_point.cpu().detach().numpy()[:, 3:6])
        o3d.io.write_point_cloud(f'debugging_ply/{rel_from}_{rel_to}_{rel_name}.ply', pc)


def load_full_image_data(scan_id_no_split, image_transform, augmentations=None):
    take_idx, pcd_idx = scan_id_no_split.split('_')
    with open(f'{OR_4D_DATA_ROOT_PATH}/export_holistic_take{take_idx}_processed/timestamp_to_pcd_and_frames_list.json') as f:
        timestamp_to_pcd_and_frames_list = json.load(f)
    images = []
    for c_idx in range(1, 7):
        color_idx_str = timestamp_to_pcd_and_frames_list[int(pcd_idx)][1][f'color_{c_idx}']
        color_path = Path(f'{OR_4D_DATA_ROOT_PATH}/export_holistic_take{take_idx}_processed/colorimage/camera0{c_idx}_colorimage-{color_idx_str}.jpg')
        img = Image.open(color_path).convert('RGB')
        if augmentations is not None:
            img = augmentations(img)
        img = image_transform(img)
        images.append(img)

    return torch.stack(images)
