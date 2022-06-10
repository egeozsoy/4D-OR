# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # specify which GPU(s) to be used

import argparse
import pprint

# noinspection PyUnresolvedReferences
from tools import _init_paths
import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
from tqdm import tqdm

# noinspection PyUnresolvedReferences
from higherhrnet_lib import models
from higherhrnet_lib.config import cfg
from higherhrnet_lib.config import check_config
from higherhrnet_lib.config import update_config
from higherhrnet_lib.core.group import HeatmapParser
from higherhrnet_lib.core.inference import aggregate_results
from higherhrnet_lib.core.inference import get_multi_stage_outputs
from higherhrnet_lib.dataset import make_test_dataloader
from higherhrnet_lib.fp16_utils.fp16util import network_to_half
from higherhrnet_lib.utils.transforms import get_final_preds
from higherhrnet_lib.utils.transforms import get_multi_scale_size
from higherhrnet_lib.utils.transforms import resize_align_multi_scale
from higherhrnet_lib.utils.utils import create_logger
from higherhrnet_lib.utils.utils import get_model_summary
# noinspection PyUnresolvedReferences
from higherhrnet_lib.utils.vis import save_debug_images
from higherhrnet_lib.utils.vis import save_valid_image

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


# markdown format output
def _print_name_value(logger, name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values + 1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )


def main():
    args = parse_args()
    update_config(cfg, args)
    check_config(cfg)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid'
    )

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )

    dump_input = torch.rand(
        (1, 3, cfg.DATASET.INPUT_SIZE, cfg.DATASET.INPUT_SIZE)
    )
    logger.info(get_model_summary(model, dump_input, verbose=cfg.VERBOSE))

    if cfg.FP16.ENABLED:
        model = network_to_half(model)

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'model_best.pth.tar'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model.cuda()
    model.eval()

    data_loader, test_dataset = make_test_dataloader(cfg)

    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )

    parser = HeatmapParser(cfg)
    all_preds = {}
    all_scores = []

    pbar = tqdm(total=len(test_dataset))
    for i, (images, image_dict) in enumerate(data_loader):
        assert 1 == images.size(0), 'Test batch size should be 1'

        image = images[0].cpu().numpy()
        # size at scale 1.0
        base_size, center, scale = get_multi_scale_size(
            image, cfg.DATASET.INPUT_SIZE, 1.0, min(cfg.TEST.SCALE_FACTOR)
        )

        with torch.no_grad():
            final_heatmaps = None
            tags_list = []
            for idx, s in enumerate(sorted(cfg.TEST.SCALE_FACTOR, reverse=True)):
                input_size = cfg.DATASET.INPUT_SIZE
                image_resized, center, scale = resize_align_multi_scale(
                    image, input_size, s, min(cfg.TEST.SCALE_FACTOR)
                )
                image_resized = transforms(image_resized)
                image_resized = image_resized.unsqueeze(0).cuda()

                outputs, heatmaps, tags = get_multi_stage_outputs(
                    cfg, model, image_resized, cfg.TEST.FLIP_TEST,
                    cfg.TEST.PROJECT2IMAGE, base_size
                )

                final_heatmaps, tags_list = aggregate_results(
                    cfg, s, final_heatmaps, tags_list, heatmaps, tags
                )

            final_heatmaps = final_heatmaps / float(len(cfg.TEST.SCALE_FACTOR))
            tags = torch.cat(tags_list, dim=4)
            grouped, scores = parser.parse(
                final_heatmaps, tags, cfg.TEST.ADJUST, cfg.TEST.REFINE
            )

            final_results = get_final_preds(
                grouped, center, scale,
                [final_heatmaps.size(3), final_heatmaps.size(2)]
            )

        pbar.update()
        identifier = f'{image_dict["take_idx"][0]}_{image_dict["image_name"][0]}'
        prefix = os.path.join(final_output_dir, identifier)
        save_valid_image(image, final_results, '{}'.format(prefix), dataset=test_dataset.name)
        assert identifier not in all_preds
        all_preds[identifier] = final_results
        all_scores.append(scores)

    pbar.close()

    np.savez_compressed(f'pred_OR_4D_hrnet_coco_{cfg.DATASET.TEST}.npz', all_preds)


if __name__ == '__main__':
    main()
