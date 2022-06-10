'''
This should easily support only processing 2D, 3D, both, parts or full, using different encoders, inputs, etc.

'''

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
from itertools import chain

from sklearn.metrics import classification_report

if __name__ == '__main__' and __package__ is None:
    from os import sys

    sys.path.append('../')
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from scene_graph_prediction.scene_graph_helpers.model.pointnets.network_PointNet import PointNetCls, PointNetRelCls, PointNetfeat
from scene_graph_prediction.scene_graph_helpers.model.pointnets.network_PointNet2 import PointNetfeat as PointNetfeat2
from scene_graph_prediction.scene_graph_helpers.model.model_utils import get_image_model
from scene_graph_prediction.scene_graph_helpers.model.gcns.network_TripletGCN import TripletGCNModel
from copy import deepcopy
from time import time


class SGPNModelWrapper(pl.LightningModule):
    def __init__(self, config, num_class, num_rel, weights_obj, weights_rel, relationNames):
        super().__init__()
        self.config = config
        self.mconfig = config['MODEL']
        self.n_object_types = 6
        self.weights_obj = weights_obj
        self.weights_rel = weights_rel
        self.relationNames = relationNames
        self.lr = float(self.config['LR'])
        # evaluation metrics
        self.train_take_rel_preds = defaultdict(list)
        self.train_take_rel_gts = defaultdict(list)
        self.val_take_rel_preds = defaultdict(list)
        self.val_take_rel_gts = defaultdict(list)
        self.reset_metrics()

        self.obj_encoder = PointNetfeat2(input_dim=6, out_size=self.mconfig['point_feature_size'], input_dropout=self.mconfig['INPUT_DROPOUT'])
        self.rel_encoder = PointNetfeat2(input_dim=7, out_size=self.mconfig['edge_feature_size'], input_dropout=self.mconfig['INPUT_DROPOUT'])
        if self.config['IMAGE_INPUT'] == 'full':
            self.full_image_model, _ = get_image_model(model_config=self.mconfig)
            # Freeze the whole model
            for param in self.full_image_model.parameters():
                param.requires_grad = False
            # Unfreeze conv head
            for param in chain(self.full_image_model.conv_head.parameters()):
                param.requires_grad = True
            self.full_image_feature_reduction = nn.Linear(self.full_image_model.num_features, self.mconfig['FULL_IMAGE_EMBEDDING_SIZE'] // 6)

        self.gcn = TripletGCNModel(num_layers=self.mconfig['N_LAYERS'],
                                   dim_node=self.mconfig['point_feature_size'],
                                   dim_edge=self.mconfig['edge_feature_size'],
                                   dim_hidden=self.mconfig['gcn_hidden_feature_size'])

        # node feature classifier
        self.obj_predictor = PointNetCls(num_class, in_size=self.mconfig['point_feature_size'],
                                         batch_norm=False, drop_out=True)
        rel_in_size = self.mconfig['edge_feature_size']
        self.rel_predictor = PointNetRelCls(
            num_rel,
            in_size=rel_in_size,
            batch_norm=False, drop_out=True, image_embedding_size=self.mconfig['FULL_IMAGE_EMBEDDING_SIZE'] if self.config['IMAGE_INPUT'] == 'full' else None,
            n_object_types=self.n_object_types)

    def freeze_image_model_batchnorm(self):
        models_to_freeze = []
        if self.config['IMAGE_INPUT'] == 'full':
            models_to_freeze.append(self.full_image_model)
        for image_model in models_to_freeze:
            for module in image_model.modules():
                if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                    if hasattr(module, 'weight'):
                        module.weight.requires_grad_(False)
                    if hasattr(module, 'bias'):
                        module.bias.requires_grad_(False)
                    module.eval()

    def forward(self, batch, return_meta_data=False):
        obj_feature = self.obj_encoder(batch['obj_points'])
        rel_feature = self.rel_encoder(batch['rel_points'])

        probs = None
        gcn_obj_feature, gcn_rel_feature = self.gcn(obj_feature, rel_feature, batch['edge_indices'])

        if self.mconfig['OBJ_PRED_FROM_GCN']:
            obj_cls = self.obj_predictor(gcn_obj_feature)
        else:
            obj_cls = self.obj_predictor(obj_feature)
        if self.config['IMAGE_INPUT'] == 'full':
            self.freeze_image_model_batchnorm()
            image_features = self.full_image_model(batch['full_image'])
            image_features = self.full_image_feature_reduction(image_features).flatten()
            rel_cls = self.rel_predictor(gcn_rel_feature, relation_objects_one_hot=batch['relation_objects_one_hot'], image_embeddings=image_features)
        else:
            rel_cls = self.rel_predictor(gcn_rel_feature, relation_objects_one_hot=batch['relation_objects_one_hot'])

        if return_meta_data:
            return obj_cls, rel_cls, obj_feature, rel_feature, gcn_obj_feature, gcn_rel_feature, probs
        else:
            return obj_cls, rel_cls

    def reset_metrics(self, split=None):
        if split == 'train':
            self.train_take_rel_preds = defaultdict(list)
            self.train_take_rel_gts = defaultdict(list)
        elif split == 'val':
            self.val_take_rel_preds = defaultdict(list)
            self.val_take_rel_gts = defaultdict(list)
        else:
            self.train_take_rel_preds = defaultdict(list)
            self.train_take_rel_gts = defaultdict(list)
            self.val_take_rel_preds = defaultdict(list)
            self.val_take_rel_gts = defaultdict(list)

    def update_metrics(self, batch, rel_pred, split='train'):
        if split == 'train':
            self.train_take_rel_preds[batch['take_idx']].extend(rel_pred.detach().cpu().numpy().argmax(1))
            self.train_take_rel_gts[batch['take_idx']].extend(batch['gt_rels'].detach().cpu().numpy())
        elif split == 'val':
            self.val_take_rel_preds[batch['take_idx']].extend(rel_pred.detach().cpu().numpy().argmax(1))
            self.val_take_rel_gts[batch['take_idx']].extend(batch['gt_rels'].detach().cpu().numpy())
        else:
            raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        if batch_idx % 100 == 0:
            print(f'Training step: {batch_idx}')
        obj_pred, rel_pred, _, _, _, _, probs = self(batch, return_meta_data=True)

        loss_obj = F.nll_loss(obj_pred, batch['gt_class'], weight=self.weights_obj.to(batch['gt_class'].device))
        loss_rel = F.nll_loss(rel_pred, batch['gt_rels'], weight=self.weights_rel.to(batch['gt_rels'].device))
        loss = self.mconfig['lambda_o'] * loss_obj + loss_rel

        self.update_metrics(batch, rel_pred, split='train')

        return loss

    def validation_step(self, batch, batch_idx):
        obj_pred, rel_pred, _, _, _, _, probs = self(batch, return_meta_data=True)
        loss_obj = F.nll_loss(obj_pred, batch['gt_class'], weight=self.weights_obj.to(batch['gt_class'].device))
        loss_rel = F.nll_loss(rel_pred, batch['gt_rels'], weight=self.weights_rel.to(batch['gt_rels'].device))
        loss = self.mconfig['lambda_o'] * loss_obj + loss_rel

        self.update_metrics(batch, rel_pred, split='val')

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        obj_pred, rel_pred, _, _, _, _, probs = self(batch, return_meta_data=True)
        predicted_relations = torch.max(rel_pred.detach(), 1)[1]
        all_scores = F.softmax(rel_pred, dim=1)

        # Get the scores that correspond to predicted_relations
        # scores = all_scores[range(rel_pred.shape[0]), predicted_relations]
        relations = []
        for idy, (edge, rel) in enumerate(zip(batch['edge_indices'].transpose(0, 1), predicted_relations)):
            if rel == self.relationNames.index('none'):
                continue
            start = edge[0]
            end = edge[1]
            start_name = batch['objs_json'][start.item() + 1]
            end_name = batch['objs_json'][end.item() + 1]
            rel_name = self.relationNames[rel]
            # print(f'{start_name} -> {rel_name} -> {end_name}')
            # if output_scores: relations.append((start_name, rel_name, end_name, scores[idy].item()))
            relations.append((start_name, rel_name, end_name))

        return (batch['scan_id'], relations)

    # def test_step(self, batch, batch_idx): # not for inference
    #     return self.validation_step(batch, batch_idx)

    def training_epoch_end(self, outputs):
        loss = sum(([i['loss'] for i in outputs]))
        self.evaluate_predictions(loss, 'train')
        self.reset_metrics(split='train')

    def validation_epoch_end(self, outputs):
        loss = sum(outputs)
        self.evaluate_predictions(loss, 'val')
        self.reset_metrics(split='val')

    # def test_epoch_end(self, outputs):
    #     return self.validation_epoch_end(outputs)

    def evaluate_predictions(self, epoch_loss, split):
        if split == 'train':
            take_rel_preds = self.train_take_rel_preds
            take_rel_gts = self.train_take_rel_gts
        elif split == 'val':
            take_rel_preds = self.val_take_rel_preds
            take_rel_gts = self.val_take_rel_gts
        else:
            raise NotImplementedError()

        self.log(f'Epoch_Loss/{split}', epoch_loss)
        all_rel_gts = []
        all_rel_preds = []
        for take_idx in sorted(take_rel_preds.keys()):
            rel_preds = take_rel_preds[take_idx]
            rel_gts = take_rel_gts[take_idx]
            all_rel_gts.extend(rel_gts)
            all_rel_preds.extend(rel_preds)
            cls_report = classification_report(rel_gts, rel_preds, labels=list(range(len(self.relationNames))),
                                               target_names=self.relationNames, output_dict=True)
            for rel_name in self.relationNames:
                for score_type in ['precision', 'recall', 'f1-score']:
                    self.log(f'{rel_name}/{take_idx}_{score_type[:2].upper()}', cls_report[rel_name][score_type])

            cls_report = classification_report(rel_gts, rel_preds, labels=list(range(len(self.relationNames))),
                                               target_names=self.relationNames)
            print(f'\nTake {take_idx}\n')
            print(cls_report)

        results = classification_report(all_rel_gts, all_rel_preds, labels=list(range(len(self.relationNames))),
                                        target_names=self.relationNames, output_dict=True)
        macro_f1 = results['macro avg']['f1-score']
        self.log(f'Epoch_Macro/{split}_PREC', results['macro avg']['precision'])
        self.log(f'Epoch_Macro/{split}_REC', results['macro avg']['recall'])
        self.log(f'Epoch_Macro/{split}_F1', results['macro avg']['f1-score'])
        self.log(f'Epoch_Micro/{split}_PREC', results['weighted avg']['precision'])
        self.log(f'Epoch_Micro/{split}_REC', results['weighted avg']['recall'])
        self.log(f'Epoch_Micro/{split}_F1', results['weighted avg']['f1-score'])
        print(f'{split} Results:\n')
        cls_report = classification_report(all_rel_gts, all_rel_preds, labels=list(range(len(self.relationNames))),
                                           target_names=self.relationNames)
        self.logger.experiment.add_text(f'Classification_Report/{split}', cls_report, self.current_epoch)
        print(cls_report)
        return macro_f1

    def configure_optimizers(self):
        optimizer = optim.AdamW(params=self.parameters(), lr=self.lr, weight_decay=float(self.config['W_DECAY']))
        return optimizer
