# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import copy
import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tqdm import tqdm
from pathlib import Path

# noinspection PyUnresolvedReferences
from voxelpose_lib import dataset, models
from voxelpose_lib.core.config import config
from voxelpose_lib.core.config import update_config
from voxelpose_lib.utils.utils import create_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args


def main():
    args = parse_args()
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'eval_map')
    cfg_name = os.path.basename(args.cfg).split('.')[0]

    gpus = [int(i) for i in config.GPUS.split(',')]
    print('=> Loading data ..')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), inference=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    print('=> Constructing models ..')
    model = eval('models.' + config.MODEL + '.get_multi_person_pose_net')(
        config, is_train=True)
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    test_model_file = os.path.join(final_output_dir, config.TEST.MODEL_FILE)
    print('=> Loading model from {}'.format(test_model_file))
    if config.TEST.MODEL_FILE and os.path.isfile(test_model_file):
        logger.info('=> load models state {}'.format(test_model_file))
        model.module.load_state_dict(torch.load(test_model_file))
    else:
        raise ValueError('Check the model file for testing!')

    model.eval()
    preds = []

    save_dict = Path(f'data/{config.DATASET.TRAIN_DATASET}_outputs')  # _GT for ground truth output
    if not save_dict.exists():
        save_dict.mkdir()

    with torch.no_grad():
        for i, (inputs, _, _, _, meta, input_heatmap) in enumerate(tqdm(test_loader)):
            frame_id = meta[0]['pcd_idx_str'][0]
            take_idx = meta[0]['take_idx'][0].item()
            # valid_gt_mask = meta[0]['joints_3d'][0].sum([1,2]) > 0.01
            # if torch.sum(valid_gt_mask) == 0:
            #     continue
            # gt_joints =  meta[0]['joints_3d'][0][valid_gt_mask].numpy().astype(np.float32)
            # np.save(str(save_dict / f'pred_{frame_id}.npy'), gt_joints)
            # continue
            pred, _, _, _, _, _ = model(meta=meta, input_heatmaps=input_heatmap)

            pred = pred.detach().cpu().numpy()[0]

            pred = pred.copy()
            pred = pred[pred[:, 0, 3] >= 0, :, :3]
            if len(pred) == 0:
                continue
            np.save(str(save_dict / f'pred_{take_idx}_{frame_id}.npy'), pred)

            for b in range(pred.shape[0]):
                preds.append(pred[b])


if __name__ == "__main__":
    import os

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used
    main()
