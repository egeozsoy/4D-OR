# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pprint

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tensorboardX import SummaryWriter

# noinspection PyUnresolvedReferences
from higherhrnet_lib import models
from higherhrnet_lib.config import cfg
from higherhrnet_lib.config import update_config
from higherhrnet_lib.core.loss import MultiLossFactory
from higherhrnet_lib.core.trainer import do_train
from higherhrnet_lib.dataset import make_dataloader
from higherhrnet_lib.fp16_utils.fp16_optimizer import FP16_Optimizer
from higherhrnet_lib.utils.utils import create_logger
from higherhrnet_lib.utils.utils import get_optimizer
from higherhrnet_lib.utils.utils import save_checkpoint
from higherhrnet_lib.utils.utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # distributed training
    parser.add_argument('--gpu',
                        help='gpu id for multiprocessing training',
                        type=str)
    parser.add_argument('--world-size',
                        default=1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist-url',
                        default='tcp://127.0.0.1:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    cfg.defrost()
    cfg.RANK = args.rank
    cfg.freeze()

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train'
    )

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    ngpus_per_node = torch.cuda.device_count()
    # Simply call main_worker function
    main_worker(
        ','.join([str(i) for i in cfg.GPUS]),
        ngpus_per_node,
        args,
        final_output_dir,
        tb_log_dir
    )


# noinspection PyUnresolvedReferences
def main_worker(
        gpu, ngpus_per_node, args, final_output_dir, tb_log_dir
):
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    update_config(cfg, args)

    # setup logger
    logger, _ = setup_logger(final_output_dir, args.rank, 'train')

    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=True
    )
    if 'OR_4D' in cfg.DATASET.DATASET:
        state_dict = torch.load('models/pytorch/pose_coco/pose_higher_hrnet_w32_512.pth')
        skip_keys = ['final_layers.0.weight', 'final_layers.0.bias', 'final_layers.1.weight', 'final_layers.1.bias', 'deconv_layers.0.0.0.weight']
        for skip_key in skip_keys:
            del state_dict[skip_key]
        model.load_state_dict(state_dict, strict=False)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    loss_factory = MultiLossFactory(cfg).cuda()

    # Data loading code
    train_loader = make_dataloader(
        cfg, is_train=True, distributed=False
    )
    logger.info(train_loader.dataset)

    best_perf = -1
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)

    if cfg.FP16.ENABLED:
        optimizer = FP16_Optimizer(
            optimizer,
            static_loss_scale=cfg.FP16.STATIC_LOSS_SCALE,
            dynamic_loss_scale=cfg.FP16.DYNAMIC_LOSS_SCALE
        )

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth.tar')
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        # train one epoch
        do_train(cfg, model, train_loader, loss_factory, optimizer, epoch,
                 final_output_dir, tb_log_dir, writer_dict, fp16=cfg.FP16.ENABLED)

        # In PyTorch 1.1.0 and later, you should call `lr_scheduler.step()` after `optimizer.step()`.
        lr_scheduler.step()

        perf_indicator = epoch
        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        if not cfg.MULTIPROCESSING_DISTRIBUTED or (
                cfg.MULTIPROCESSING_DISTRIBUTED
                and args.rank == 0
        ):
            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state{}.pth.tar'.format(gpu)
    )

    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    import os

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used
    main()
