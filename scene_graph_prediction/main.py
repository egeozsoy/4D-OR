import warnings

warnings.filterwarnings('ignore')
import argparse
from pathlib import Path

import json_tricks as json  # Allows to load integers etc. correctly
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from scene_graph_prediction.scene_graph_helpers.dataset.or_dataset import ORDataset
from scene_graph_prediction.scene_graph_helpers.model.scene_graph_prediction_model import SGPNModelWrapper
from pytorch_lightning.callbacks import ModelCheckpoint


def config_loader(config_path: str):
    config_path = Path('scene_graph_prediction/scene_graph_helpers/configs') / config_path
    with open(config_path, 'r') as f:
        config = json.load(f, ignore_comments=True)
    return config


def find_checkpoint_path(log_dir: str):
    def epoch_int(file_str):
        return int(file_str.split('=')[1].replace('.ckpt', ''))

    log_dir = Path(log_dir)
    checkpoint_folder = log_dir / 'checkpoints'
    checkpoints = sorted(checkpoint_folder.glob('*.ckpt'), key=lambda x: epoch_int(x.name), reverse=True)
    if len(checkpoints) == 0:
        return None
    return checkpoints[0]


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='example.json', help='configuration file name. Relative path under given path')
    args = parser.parse_args()
    pl.seed_everything(42, workers=True)
    config = config_loader(args.config)
    mode = 'train'  # can be train/evaluate/infer
    paper_weight = None  # can also be 'scene_graph_prediction/scene_graph_helpers/paper_weights/paper_model_no_gt_no_images.pth' or 'scene_graph_prediction/scene_graph_helpers/paper_weights/paper_model_no_gt_with_images.pth'

    name = args.config.replace('.json', '')

    logger = pl.loggers.TensorBoardLogger('scene_graph_prediction/scene_graph_helpers/logs', name=name, version=0)
    checkpoint_path = find_checkpoint_path(logger.log_dir)

    if mode == 'train':
        train_dataset = ORDataset(config, 'train', shuffle_objs=True)
        val_dataset = ORDataset(config, 'val')

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=config['NUM_WORKERS'], pin_memory=True,
                                  collate_fn=train_dataset.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=config['NUM_WORKERS'], pin_memory=True, collate_fn=val_dataset.collate_fn)

        model = SGPNModelWrapper(config, num_class=len(val_dataset.classNames), num_rel=len(val_dataset.relationNames), weights_obj=train_dataset.w_cls_obj,
                                 weights_rel=train_dataset.w_cls_rel, relationNames=train_dataset.relationNames)

        checkpoint = ModelCheckpoint(filename='{epoch}', save_top_k=-1, every_n_epochs=1)
        trainer = pl.Trainer(gpus=1, max_epochs=config['MAX_EPOCHES'], logger=logger, check_val_every_n_epoch=1, log_every_n_steps=50, num_sanity_val_steps=0,
                             callbacks=[checkpoint, pl.callbacks.progress.RichProgressBar()], benchmark=False,
                             precision=16)
        print('Start Training')
        trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)

    elif mode == 'evaluate':
        train_dataset = ORDataset(config, 'train', shuffle_objs=True)
        eval_dataset = ORDataset(config, 'val')
        eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=config['NUM_WORKERS'], pin_memory=True,
                                 collate_fn=eval_dataset.collate_fn)

        if paper_weight is not None:  # Use to replicate paper results
            model = SGPNModelWrapper(config, num_class=len(eval_dataset.classNames), num_rel=len(eval_dataset.relationNames),
                                     weights_obj=train_dataset.w_cls_obj,
                                     weights_rel=train_dataset.w_cls_rel, relationNames=train_dataset.relationNames)
            model.load_state_dict(torch.load(paper_weight))
            checkpoint_path = None
        else:
            # checkpoint_path = None # Can hardcode a difference checkpoint path here if wanted
            model = SGPNModelWrapper.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config, num_class=len(eval_dataset.classNames),
                                                          num_rel=len(eval_dataset.relationNames), weights_obj=train_dataset.w_cls_obj,
                                                          weights_rel=train_dataset.w_cls_rel, relationNames=train_dataset.relationNames)

        trainer = pl.Trainer(gpus=1, max_epochs=config['MAX_EPOCHES'], logger=logger, check_val_every_n_epoch=1, log_every_n_steps=50, num_sanity_val_steps=0,
                             callbacks=[pl.callbacks.progress.RichProgressBar()])
        assert checkpoint_path is not None or paper_weight is not None
        trainer.validate(model, eval_loader, ckpt_path=checkpoint_path)
    elif mode == 'infer':
        train_dataset = ORDataset(config, 'train', shuffle_objs=True)
        infer_split = 'test'
        eval_dataset = ORDataset(config, infer_split, for_eval=True)
        eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=config['NUM_WORKERS'], pin_memory=True,
                                 collate_fn=eval_dataset.collate_fn)

        if paper_weight is not None:
            model = SGPNModelWrapper(config, num_class=len(eval_dataset.classNames), num_rel=len(eval_dataset.relationNames),
                                     weights_obj=train_dataset.w_cls_obj,
                                     weights_rel=train_dataset.w_cls_rel, relationNames=train_dataset.relationNames)
            model.load_state_dict(torch.load(paper_weight))
            checkpoint_path = None
        else:
            # checkpoint_path = None # Can hardcode a difference checkpoint path here if wanted
            model = SGPNModelWrapper.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config, num_class=len(eval_dataset.classNames),
                                                          num_rel=len(eval_dataset.relationNames), weights_obj=train_dataset.w_cls_obj,
                                                          weights_rel=train_dataset.w_cls_rel, relationNames=train_dataset.relationNames)
        trainer = pl.Trainer(gpus=1, max_epochs=config['MAX_EPOCHES'], logger=logger, check_val_every_n_epoch=1, log_every_n_steps=50, num_sanity_val_steps=0,
                             callbacks=[pl.callbacks.progress.RichProgressBar()])
        assert checkpoint_path is not None or paper_weight is not None
        results = trainer.predict(model, eval_loader, ckpt_path=checkpoint_path)
        scan_relations = {key: value for key, value in results}
        output_name = f'scan_relations_{name}_{infer_split}.json'
        with open(output_name, 'w') as f:
            json.dump(scan_relations, f)
    else:
        raise ValueError('Invalid mode')


if __name__ == '__main__':
    import subprocess

    subprocess.call(['nvidia-smi', '-L'])
    main()
