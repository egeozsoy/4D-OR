import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used
import time
import numpy as np
import json
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from torch.utils.tensorboard import SummaryWriter
from external_src.group_free_3D.utils import get_scheduler
from external_src.group_free_3D.models import GroupFreeDetector, get_loss
from external_src.group_free_3D.models import APCalculator, parse_predictions, parse_groundtruths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR


def parse_option():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--width', default=1, type=int, help='backbone width')
    parser.add_argument('--num_target', type=int, default=256, help='Proposal number [default: 256]')
    parser.add_argument('--sampling', default='kps', type=str, help='Query points sampling method (kps, fps)')

    # Transformer
    parser.add_argument('--nhead', default=8, type=int, help='multi-head number')
    parser.add_argument('--num_decoder_layers', default=6, type=int, help='number of decoder layers')
    parser.add_argument('--dim_feedforward', default=2048, type=int, help='dim_feedforward')
    parser.add_argument('--transformer_dropout', default=0.1, type=float, help='transformer_dropout')
    parser.add_argument('--transformer_activation', default='relu', type=str, help='transformer_activation')
    parser.add_argument('--self_position_embedding', default='loc_learned', type=str,
                        help='position_embedding in self attention (none, xyz_learned, loc_learned)')
    parser.add_argument('--cross_position_embedding', default='xyz_learned', type=str,
                        help='position embedding in cross attention (none, xyz_learned)')

    # Loss
    parser.add_argument('--query_points_generator_loss_coef', default=0.8, type=float)
    parser.add_argument('--obj_loss_coef', default=0.1, type=float, help='Loss weight for objectness loss')
    parser.add_argument('--box_loss_coef', default=1, type=float, help='Loss weight for box loss')
    parser.add_argument('--sem_cls_loss_coef', default=0.1, type=float, help='Loss weight for classification loss')
    parser.add_argument('--center_loss_type', default='smoothl1', type=str, help='(smoothl1, l1)')
    parser.add_argument('--center_delta', default=1.0, type=float, help='delta for smoothl1 loss in center loss')  # TODO maybe back to 0.4
    parser.add_argument('--size_loss_type', default='smoothl1', type=str, help='(smoothl1, l1)')
    parser.add_argument('--size_delta', default=1.0, type=float, help='delta for smoothl1 loss in size loss')  # TODO maybe back to 0.1
    parser.add_argument('--heading_loss_type', default='smoothl1', type=str, help='(smoothl1, l1)')
    parser.add_argument('--heading_delta', default=1.0, type=float, help='delta for smoothl1 loss in heading loss')
    parser.add_argument('--query_points_obj_topk', default=4, type=int, help='query_points_obj_topk')
    parser.add_argument('--size_cls_agnostic', action='store_true', help='Use class-agnostic size prediction.')

    # Data
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 8]')  # one gpu 8 two gpus 16
    parser.add_argument('--dataset', default='OR_4D', help='Dataset name. sunrgbd or scannet. [default: scannet]')
    parser.add_argument('--num_point', type=int, default=200000, help='Point Number [default: 50000]')  # maybe use 50000
    parser.add_argument('--data_root', default='data', help='data root path')
    parser.add_argument('--use_height', action='store_true', help='Use height signal in input.')
    parser.add_argument('--use_sunrgbd_v2', action='store_true', help='Use V2 box labels for SUN RGB-D dataset')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')

    # Training
    parser.add_argument('--start_epoch', type=int, default=1, help='Epoch to run [default: 1]')
    parser.add_argument('--max_epoch', type=int, default=180, help='Epoch to run [default: 180]')  # TODO maybe 180 or 400
    parser.add_argument('--optimizer', type=str, default='adamW', help='optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Optimization L2 weight decay [default: 0.0005]')
    parser.add_argument('--learning_rate', type=float, default=0.004,
                        help='Initial learning rate for all except decoder [default: 0.004]')
    parser.add_argument('--decoder_learning_rate', type=float, default=0.0004,
                        help='Initial learning rate for decoder [default: 0.0004]')
    parser.add_argument('--lr-scheduler', type=str, default='step',
                        choices=["step", "cosine"], help="learning rate scheduler")
    parser.add_argument('--warmup-epoch', type=int, default=-1, help='warmup epoch')
    parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--lr_decay_epochs', type=int, default=[100, 140], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')  # TODO [100, 140] or [280,340]
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--clip_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--bn_momentum', type=float, default=0.1, help='Default bn momeuntum')
    parser.add_argument('--syncbn', action='store_true', help='whether to use sync bn')

    # io
    parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
    parser.add_argument('--log_dir', default='logroot', help='Dump dir to save model checkpoint [default: log]')
    parser.add_argument('--print_freq', type=int, default=40, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--val_freq', type=int, default=40, help='val frequency')

    # others
    parser.add_argument('--ap_iou_thresholds', type=float, default=[0.25, 0.5], nargs='+',
                        help='A list of AP IoU thresholds [default: 0.25,0.5]')
    parser.add_argument("--rng_seed", type=int, default=0, help='manual seed')

    args, unparsed = parser.parse_known_args()

    return args


def load_checkpoint(args, model, optimizer, scheduler):
    print("=> loading checkpoint '{}'".format(args.checkpoint_path))

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    print("=> loaded successfully '{}' (epoch {})".format(args.checkpoint_path, checkpoint['epoch']))

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(args, epoch, model, optimizer, scheduler, save_cur=False):
    print('==> Saving...')
    state = {
        'config': args,
        'save_path': '',
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }

    if save_cur:
        state['save_path'] = os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')
        torch.save(state, os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth'))
        print("Saved in {}".format(os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')))
    elif epoch % args.save_freq == 0:
        state['save_path'] = os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')
        torch.save(state, os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth'))
        print("Saved in {}".format(os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')))
    else:
        # state['save_path'] = 'current.pth'
        # torch.save(state, os.path.join(args.log_dir, 'current.pth'))
        print("not saving checkpoint")
        pass


def get_loader(args):
    if args.dataset == 'OR_4D':
        from external_src.group_free_3D.OR_4D.OR_4D_detection_dataset import OR_4DDetectionDataset
        from external_src.group_free_3D.OR_4D.model_util_OR_4D import OR_4DDatasetConfig

        DATASET_CONFIG = OR_4DDatasetConfig()
        TRAIN_DATASET = OR_4DDetectionDataset('train', num_points=args.num_point,
                                              augment=True,
                                              use_color=True,
                                              use_height=True if args.use_height else False,
                                              data_root=args.data_root)
        TEST_DATASET = OR_4DDetectionDataset('val', num_points=args.num_point,
                                             augment=False,
                                             use_color=True,
                                             use_height=True if args.use_height else False,
                                             data_root=args.data_root)

    else:
        raise NotImplementedError(f'Unknown dataset {args.dataset}. Exiting...')

    print(f"train_len: {len(TRAIN_DATASET)}, test_len: {len(TEST_DATASET)}")

    train_loader = torch.utils.data.DataLoader(TRAIN_DATASET,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               drop_last=True)

    test_loader = torch.utils.data.DataLoader(TEST_DATASET,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              drop_last=False)
    print(f"train_loader_len: {len(train_loader)}, test_loader_len: {len(test_loader)}")

    return train_loader, test_loader, DATASET_CONFIG


def get_model(args, DATASET_CONFIG):
    if args.use_height:
        num_input_channel = 4
    else:
        num_input_channel = 3
    model = GroupFreeDetector(num_class=DATASET_CONFIG.num_class,
                              num_heading_bin=DATASET_CONFIG.num_heading_bin,
                              num_size_cluster=DATASET_CONFIG.num_size_cluster,
                              mean_size_arr=DATASET_CONFIG.mean_size_arr,
                              input_feature_dim=num_input_channel,
                              width=args.width,
                              bn_momentum=args.bn_momentum,
                              sync_bn=True if args.syncbn else False,
                              num_proposal=args.num_target,
                              sampling=args.sampling,
                              dropout=args.transformer_dropout,
                              activation=args.transformer_activation,
                              nhead=args.nhead,
                              num_decoder_layers=args.num_decoder_layers,
                              dim_feedforward=args.dim_feedforward,
                              self_position_embedding=args.self_position_embedding,
                              cross_position_embedding=args.cross_position_embedding,
                              size_cls_agnostic=True if args.size_cls_agnostic else False)

    criterion = get_loss
    return model, criterion


def main(args):
    train_loader, test_loader, DATASET_CONFIG = get_loader(args)
    n_data = len(train_loader.dataset)
    print(f"length of training dataset: {n_data}")
    n_data = len(test_loader.dataset)
    print(f"length of testing dataset: {n_data}")

    model, criterion = get_model(args, DATASET_CONFIG)
    # optimizer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "decoder" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "decoder" in n and p.requires_grad],
            "lr": args.decoder_learning_rate,
        },
    ]
    optimizer = optim.AdamW(param_dicts,
                            lr=args.learning_rate,
                            weight_decay=args.weight_decay)

    scheduler = get_scheduler(optimizer, len(train_loader), args)
    model = model.cuda()
    model = DataParallel(model)

    if args.checkpoint_path:
        assert os.path.isfile(args.checkpoint_path)
        load_checkpoint(args, model, optimizer, scheduler)

    # Used for AP calculation
    CONFIG_DICT = {'remove_empty_box': False, 'use_3d_nms': True,
                   'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': True,
                   'per_class_proposal': True, 'conf_thresh': 0.0,
                   'dataset_config': DATASET_CONFIG}

    for epoch in range(args.start_epoch, args.max_epoch + 1):

        tic = time.time()

        train_one_epoch(epoch, train_loader, DATASET_CONFIG, model, criterion, optimizer, scheduler, args)

        print('epoch {}, total time {:.2f}, '
              'lr_base {:.5f}, lr_decoder {:.5f}'.format(epoch, (time.time() - tic),
                                                         optimizer.param_groups[0]['lr'],
                                                         optimizer.param_groups[1]['lr']))

        if epoch % args.val_freq == 0:
            train_mAP, train_mAPs = evaluate_one_epoch(train_loader, DATASET_CONFIG, CONFIG_DICT, args.ap_iou_thresholds, model,
                                                       criterion, args, dataset_name='train')
            val_mAP, val_mAPs = evaluate_one_epoch(test_loader, DATASET_CONFIG, CONFIG_DICT, args.ap_iou_thresholds, model,
                                                   criterion, args)
        # save model
        save_checkpoint(args, epoch, model, optimizer, scheduler)
    evaluate_one_epoch(test_loader, DATASET_CONFIG, CONFIG_DICT, args.ap_iou_thresholds, model, criterion, args)
    save_checkpoint(args, 'last', model, optimizer, scheduler, save_cur=True)
    print("Saved in {}".format(os.path.join(args.log_dir, f'ckpt_epoch_last.pth')))
    return os.path.join(args.log_dir, f'ckpt_epoch_last.pth')


def train_one_epoch(epoch, train_loader, DATASET_CONFIG, model, criterion, optimizer, scheduler, config):
    global global_step
    stat_dict = {}  # collect statistics
    model.train()  # set model to training mode
    for batch_idx, batch_data_label in enumerate(train_loader):
        scan_names = batch_data_label.pop('scan_name')
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)
        inputs = {'point_clouds': batch_data_label['point_clouds']}

        # Forward pass
        end_points = model(inputs)

        # Compute loss and gradients, update parameters.
        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, DATASET_CONFIG,
                                     num_decoder_layers=config.num_decoder_layers,
                                     query_points_generator_loss_coef=config.query_points_generator_loss_coef,
                                     obj_loss_coef=config.obj_loss_coef,
                                     box_loss_coef=config.box_loss_coef,
                                     sem_cls_loss_coef=config.sem_cls_loss_coef,
                                     query_points_obj_topk=config.query_points_obj_topk,
                                     center_loss_type=config.center_loss_type,
                                     center_delta=config.center_delta,
                                     size_loss_type=config.size_loss_type,
                                     size_delta=config.size_delta,
                                     heading_loss_type=config.heading_loss_type,
                                     heading_delta=config.heading_delta,
                                     size_cls_agnostic=config.size_cls_agnostic)

        optimizer.zero_grad()
        loss.backward()
        if config.clip_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)
        optimizer.step()
        scheduler.step()

        tb_logger.add_scalar('Loss/train', loss, global_step)
        global_step += 1

        # Accumulate statistics and print out
        stat_dict['grad_norm'] = grad_total_norm
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                if isinstance(end_points[key], float):
                    stat_dict[key] += end_points[key]
                else:
                    stat_dict[key] += end_points[key].item()

        if (batch_idx + 1) % config.print_freq == 0:
            # print(f'Train: [{epoch}][{batch_idx + 1}/{len(train_loader)}]  ' + ''.join(
            #     [f'{key} {stat_dict[key] / config.print_freq:.4f} \t'
            #      for key in sorted(stat_dict.keys()) if 'loss' not in key]))
            # print(f"grad_norm: {stat_dict['grad_norm']}")
            # print(''.join([f'{key} {stat_dict[key] / config.print_freq:.4f} \t'
            #                for key in sorted(stat_dict.keys()) if
            #                'loss' in key and 'proposal_' not in key and 'last_' not in key and 'head_' not in key]))
            # print(''.join([f'{key} {stat_dict[key] / config.print_freq:.4f} \t'
            #                for key in sorted(stat_dict.keys()) if 'last_' in key]))
            # print(''.join([f'{key} {stat_dict[key] / config.print_freq:.4f} \t'
            #                for key in sorted(stat_dict.keys()) if 'proposal_' in key]))
            # for ihead in range(config.num_decoder_layers - 2, -1, -1):
            #     print(''.join([f'{key} {stat_dict[key] / config.print_freq:.4f} \t'
            #                    for key in sorted(stat_dict.keys()) if f'{ihead}head_' in key]))

            for key in sorted(stat_dict.keys()):
                stat_dict[key] = 0


def evaluate_one_epoch(test_loader, DATASET_CONFIG, CONFIG_DICT, AP_IOU_THRESHOLDS, model, criterion, config, dataset_name='val'):
    print(f'Evaluating for dataset : {dataset_name}')
    stat_dict = {}

    if config.num_decoder_layers > 0:
        prefixes = ['last_', 'proposal_'] + [f'{i}head_' for i in range(config.num_decoder_layers - 1)]
    else:
        prefixes = ['proposal_']  # only proposal
    ap_calculator_list = [APCalculator(iou_thresh, DATASET_CONFIG.class2type) \
                          for iou_thresh in AP_IOU_THRESHOLDS]
    mAPs = [[iou_thresh, {k: 0 for k in prefixes}] for iou_thresh in AP_IOU_THRESHOLDS]

    model.eval()  # set model to eval mode (for bn and dp)
    batch_pred_map_cls_dict = {k: [] for k in prefixes}
    batch_gt_map_cls_dict = {k: [] for k in prefixes}
    total_val_loss = 0.
    for batch_idx, batch_data_label in enumerate(test_loader):
        scan_names = batch_data_label.pop('scan_name')
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)

        # Forward pass
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        with torch.no_grad():
            end_points = model(inputs)

        # Compute loss
        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, DATASET_CONFIG,
                                     num_decoder_layers=config.num_decoder_layers,
                                     query_points_generator_loss_coef=config.query_points_generator_loss_coef,
                                     obj_loss_coef=config.obj_loss_coef,
                                     box_loss_coef=config.box_loss_coef,
                                     sem_cls_loss_coef=config.sem_cls_loss_coef,
                                     query_points_obj_topk=config.query_points_obj_topk,
                                     center_loss_type=config.center_loss_type,
                                     center_delta=config.center_delta,
                                     size_loss_type=config.size_loss_type,
                                     size_delta=config.size_delta,
                                     heading_loss_type=config.heading_loss_type,
                                     heading_delta=config.heading_delta,
                                     size_cls_agnostic=config.size_cls_agnostic)

        total_val_loss += loss.item()

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                if isinstance(end_points[key], float):
                    stat_dict[key] += end_points[key]
                else:
                    stat_dict[key] += end_points[key].item()

        for prefix in prefixes:
            batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT, prefix,
                                                   size_cls_agnostic=config.size_cls_agnostic)
            batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT,
                                                  size_cls_agnostic=config.size_cls_agnostic)
            batch_pred_map_cls_dict[prefix].append(batch_pred_map_cls)
            batch_gt_map_cls_dict[prefix].append(batch_gt_map_cls)

    mAP = 0.0
    for prefix in prefixes:
        for (batch_pred_map_cls, batch_gt_map_cls) in zip(batch_pred_map_cls_dict[prefix],
                                                          batch_gt_map_cls_dict[prefix]):
            for ap_calculator in ap_calculator_list:
                ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
        # Evaluate average precision
        for i, ap_calculator in enumerate(ap_calculator_list):
            metrics_dict = ap_calculator.compute_metrics()
            print(f'=====================>{prefix} IOU THRESH: {AP_IOU_THRESHOLDS[i]}<=====================')
            for key in metrics_dict:
                print(f'{key} {metrics_dict[key]}')
            if prefix == 'last_' and ap_calculator.ap_iou_thresh > 0.3:
                mAP = metrics_dict['mAP']
            mAPs[i][1][prefix] = metrics_dict['mAP']
            tb_logger.add_scalar(f'mAP/{dataset_name}_{ap_calculator.ap_iou_thresh}', metrics_dict['mAP_nonzero'], global_step)
            tb_logger.add_scalar(f'AR/{dataset_name}_{ap_calculator.ap_iou_thresh}', metrics_dict['AR_nonzero'], global_step)
            ap_calculator.reset()

    if dataset_name == 'val':
        tb_logger.add_scalar('Loss/val', total_val_loss, global_step)
    return mAP, mAPs


if __name__ == '__main__':

    tb_logger = SummaryWriter()
    global_step = 0

    opt = parse_option()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    LOG_DIR = os.path.join(opt.log_dir, 'group_free',
                           f'{opt.dataset}_{int(time.time())}', f'{np.random.randint(100000000)}')
    while os.path.exists(LOG_DIR):
        LOG_DIR = os.path.join(opt.log_dir, 'group_free',
                               f'{opt.dataset}_{int(time.time())}', f'{np.random.randint(100000000)}')
    opt.log_dir = LOG_DIR
    os.makedirs(opt.log_dir, exist_ok=True)

    path = os.path.join(opt.log_dir, "config.json")
    with open(path, 'w') as f:
        json.dump(vars(opt), f, indent=2)
    print("Full config saved to {}".format(path))
    print(str(vars(opt)))

    ckpt_path = main(opt)
    a = 1
