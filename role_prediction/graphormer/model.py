# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import itertools
import json

from sklearn.metrics import classification_report

from role_prediction.graphormer.data import get_dataset
import torch
import math
import torch.nn as nn
import pytorch_lightning as pl

from role_prediction.graphormer.lr import PolynomialDecayLR
from role_prediction.graphormer.role_prediction_helpers import eval_role_prediction_perf, output_role_predictions
from helpers.configurations import OR_4D_DATA_ROOT_PATH

from role_prediction.graphormer.utils.flag import flag_bounded


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class Graphormer(pl.LightningModule):
    def __init__(
            self,
            n_layers,
            num_heads,
            hidden_dim,
            dropout_rate,
            intput_dropout_rate,
            weight_decay,
            ffn_dim,
            dataset_name,
            warmup_updates,
            tot_updates,
            peak_lr,
            end_lr,
            edge_type,
            multi_hop_max_dist,
            attention_dropout_rate,
            flag=False,
            flag_m=3,
            flag_step_size=1e-3,
            flag_mag=1e-3,
            role_prediction_save_name=None
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_heads = num_heads
        if dataset_name == 'ZINC':
            self.atom_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
            self.edge_encoder = nn.Embedding(64, num_heads, padding_idx=0)
            self.edge_type = edge_type
            if self.edge_type == 'multi_hop':
                self.edge_dis_encoder = nn.Embedding(
                    40 * num_heads * num_heads, 1)
            self.spatial_pos_encoder = nn.Embedding(40, num_heads, padding_idx=0)
            self.in_degree_encoder = nn.Embedding(
                64, hidden_dim, padding_idx=0)
            self.out_degree_encoder = nn.Embedding(
                64, hidden_dim, padding_idx=0)
        elif dataset_name == 'scene_graph' or dataset_name == 'role_prediction':
            # self.atom_encoder = nn.Embedding(10, hidden_dim, padding_idx=0)
            self.atom_encoder = nn.Embedding(30, hidden_dim, padding_idx=0)
            # self.edge_encoder = nn.Embedding(18, num_heads, padding_idx=0)
            self.edge_encoder = nn.Embedding(5, num_heads, padding_idx=0)
            self.edge_type = edge_type
            if self.edge_type == 'multi_hop':
                self.edge_dis_encoder = nn.Embedding(40 * num_heads * num_heads, 1)
            self.spatial_pos_encoder = nn.Embedding(64, num_heads, padding_idx=0)
            self.in_degree_encoder = nn.Embedding(
                64, hidden_dim, padding_idx=0)
            self.out_degree_encoder = nn.Embedding(
                64, hidden_dim, padding_idx=0)
        else:
            self.atom_encoder = nn.Embedding(
                512 * 9 + 1, hidden_dim, padding_idx=0)
            self.edge_encoder = nn.Embedding(
                512 * 3 + 1, num_heads, padding_idx=0)
            self.edge_type = edge_type
            if self.edge_type == 'multi_hop':
                self.edge_dis_encoder = nn.Embedding(
                    128 * num_heads * num_heads, 1)
            self.spatial_pos_encoder = nn.Embedding(512, num_heads, padding_idx=0)
            self.in_degree_encoder = nn.Embedding(
                512, hidden_dim, padding_idx=0)
            self.out_degree_encoder = nn.Embedding(
                512, hidden_dim, padding_idx=0)

        self.input_dropout = nn.Dropout(intput_dropout_rate)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)

        if dataset_name == 'PCQM4M-LSC':
            self.out_proj = nn.Linear(hidden_dim, 1)
        else:
            self.downstream_out_proj = nn.Linear(
                hidden_dim, get_dataset(dataset_name)['num_class'])

        self.graph_token = nn.Embedding(1, hidden_dim)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.evaluator = get_dataset(dataset_name)['evaluator']
        self.metric = get_dataset(dataset_name)['metric']
        if dataset_name == 'role_prediction':
            self.loss_fn = nn.CrossEntropyLoss()
            self.label_names = ['Patient', 'head_surgeon', 'assistant_surgeon', 'circulating_nurse', 'anaesthetist']
        else:
            self.loss_fn = get_dataset(dataset_name)['loss_fn']
        self.dataset_name = dataset_name

        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.weight_decay = weight_decay
        self.multi_hop_max_dist = multi_hop_max_dist

        self.flag = flag
        self.flag_m = flag_m
        self.flag_step_size = flag_step_size
        self.flag_mag = flag_mag
        self.hidden_dim = hidden_dim
        self.role_prediction_save_name = role_prediction_save_name
        self.automatic_optimization = not self.flag
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data, perturb=None):
        attn_bias, spatial_pos, x = batched_data.attn_bias, batched_data.spatial_pos, batched_data.x
        in_degree, out_degree = batched_data.in_degree, batched_data.in_degree
        edge_input, attn_edge_type = batched_data.edge_input, batched_data.attn_edge_type
        # graph_attn_bias
        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1)  # [n_graph, n_head, n_node+1, n_node+1]

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:,
                                        :, 1:, 1:] + spatial_pos_bias
        # reset spatial pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # edge feature
        if self.edge_type == 'multi_hop':
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
            # set 1 to 1, x > 1 to x - 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, :self.multi_hop_max_dist, :]
            # [n_graph, n_node, n_node, max_dist, n_head]
            edge_input = self.edge_encoder(edge_input).mean(-2)
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(
                3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
            edge_input_flat = torch.bmm(edge_input_flat, self.edge_dis_encoder.weight.reshape(
                -1, self.num_heads, self.num_heads)[:max_dist, :, :])
            edge_input = edge_input_flat.reshape(
                max_dist, n_graph, n_node, n_node, self.num_heads).permute(1, 2, 3, 0, 4)
            edge_input = (edge_input.sum(-2) /
                          (spatial_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)
        else:
            # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            edge_input = self.edge_encoder(
                attn_edge_type).mean(-2).permute(0, 3, 1, 2)

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:,
                                        :, 1:, 1:] + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        # node feauture + graph token
        node_feature = self.atom_encoder(x).sum(
            dim=-2)  # [n_graph, n_node, n_hidden]
        if self.flag and perturb is not None:
            node_feature += perturb

        node_feature = node_feature + \
                       self.in_degree_encoder(in_degree) + \
                       self.out_degree_encoder(out_degree)
        graph_token_feature = self.graph_token.weight.unsqueeze(
            0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat(
            [graph_token_feature, node_feature], dim=1)

        # transfomrer encoder
        output = self.input_dropout(graph_node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, graph_attn_bias)
        output = self.final_ln(output)

        # output part
        if self.dataset_name == 'PCQM4M-LSC':
            # get whole graph rep
            output = self.out_proj(output[:, 0, :])
        elif self.dataset_name == 'role_prediction':
            is_target = batched_data.meta['is_target']
            is_target = torch.cat([is_target, torch.ones_like(is_target)[:, :1]], dim=1)[:, :, 0]
            target_node_embeddings = output.flatten(0, 1)[is_target.flatten(0, 1) == 2]
            if len(target_node_embeddings) == 0:
                target_node_embeddings = torch.zeros(80, device=torch.device('cuda'), dtype=torch.float32)[None]
            else:
                target_node_embeddings = target_node_embeddings.mean(dim=0, keepdim=True)
            output = self.downstream_out_proj(target_node_embeddings)
        else:
            output = self.downstream_out_proj(output[:, 0, :])
        return output

    def training_step(self, batched_data, batch_idx):
        if self.dataset_name == 'ogbg-molpcba':
            if not self.flag:
                y_hat = self(batched_data).view(-1)
                y_gt = batched_data.y.view(-1).float()
                mask = ~torch.isnan(y_gt)
                loss = self.loss_fn(y_hat[mask], y_gt[mask])
            else:
                y_gt = batched_data.y.view(-1).float()
                mask = ~torch.isnan(y_gt)

                def forward(perturb):
                    return self(batched_data, perturb)

                model_forward = (self, forward)
                n_graph, n_node = batched_data.x.size()[:2]
                perturb_shape = (n_graph, n_node, self.hidden_dim)

                optimizer = self.optimizers()
                optimizer.zero_grad()
                loss, _ = flag_bounded(model_forward, perturb_shape, y_gt[mask], optimizer, batched_data.x.device, self.loss_fn,
                                       m=self.flag_m, step_size=self.flag_step_size, mag=self.flag_mag, mask=mask)
                self.lr_schedulers().step()

        elif self.dataset_name == 'ogbg-molhiv':
            if not self.flag:
                y_hat = self(batched_data).view(-1)
                y_gt = batched_data.y.view(-1).float()
                loss = self.loss_fn(y_hat, y_gt)
            else:
                y_gt = batched_data.y.view(-1).float()

                def forward(perturb):
                    return self(batched_data, perturb)

                model_forward = (self, forward)
                n_graph, n_node = batched_data.x.size()[:2]
                perturb_shape = (n_graph, n_node, self.hidden_dim)

                optimizer = self.optimizers()
                optimizer.zero_grad()
                loss, _ = flag_bounded(model_forward, perturb_shape, y_gt, optimizer, batched_data.x.device, self.loss_fn,
                                       m=self.flag_m, step_size=self.flag_step_size, mag=self.flag_mag)
                self.lr_schedulers().step()
        elif self.dataset_name == 'role_prediction':
            y_pred = self(batched_data)
            y_gt = batched_data.y
            loss = self.loss_fn(y_pred, y_gt)  # We can also penalize more for bigger batched data (guessing a big track wrong is worse)
            return {'loss': loss, 'y_pred': y_pred, 'y_true': y_gt}
        else:
            y_hat = self(batched_data).view(-1)
            y_gt = batched_data.y.view(-1)
            loss = self.loss_fn(y_hat, y_gt)
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def training_epoch_end(self, outputs) -> None:
        y_pred = torch.cat([i['y_pred'] for i in outputs]).argmax(1).cpu().detach().numpy()
        y_true = torch.cat([i['y_true'] for i in outputs]).cpu().detach().numpy()

        results = classification_report(y_true, y_pred, labels=list(range(len(self.label_names))),
                                        target_names=self.label_names, output_dict=True)

        self.log('train_micro_f1', results['accuracy'] if 'accuracy' in results else results['micro avg']['f1-score'])
        self.log('train_macro_f1', results['macro avg']['f1-score'])

        results = classification_report(y_true, y_pred, labels=list(range(len(self.label_names))),
                                        target_names=self.label_names)
        print('Train Results:')
        print(results)

    def validation_step(self, batched_data, batch_idx):
        if self.dataset_name in ['PCQM4M-LSC', 'ZINC']:
            y_pred = self(batched_data).view(-1)
            y_true = batched_data.y.view(-1)
        else:
            y_pred = self(batched_data)
            y_true = batched_data.y
        return {
            'y_pred': y_pred,
            'y_true': y_true,
        }

    def validation_epoch_end(self, outputs):
        y_pred = torch.cat([i['y_pred'] for i in outputs])
        y_true = torch.cat([i['y_true'] for i in outputs])
        if self.dataset_name == 'role_prediction':
            take_to_results_train, train_split_results = eval_role_prediction_perf(self.trainer.datamodule.dataset['train_dataset'], self)
            take_to_results_val, val_split_results = eval_role_prediction_perf(self.trainer.datamodule.dataset['valid_dataset'], self)
            take_to_results = {**take_to_results_train, **take_to_results_val}
            for take, results in take_to_results.items():
                self.log(f'{take}_micro_f1', results['micro_f1'])
                self.log(f'{take}_macro_f1', results['macro_f1'])

            print(f'Result For Split Train:')
            print(train_split_results)
            print(f'Result For Split Val:')
            print(val_split_results)

        if self.dataset_name == 'ogbg-molpcba':
            mask = ~torch.isnan(y_true)
            loss = self.loss_fn(y_pred[mask], y_true[mask])
            self.log('valid_ap', loss, sync_dist=True)
        elif self.dataset_name == 'role_prediction':
            y_pred = y_pred.argmax(1).cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
            results = classification_report(y_true, y_pred, labels=list(range(len(self.label_names))),
                                            target_names=self.label_names, output_dict=True)

            self.log('valid_micro_f1', results['accuracy'] if 'accuracy' in results else results['micro avg']['f1-score'])
            self.log('valid_macro_f1', results['macro avg']['f1-score'])

            results = classification_report(y_true, y_pred, labels=list(range(len(self.label_names))),
                                            target_names=self.label_names)

            print('Val Results:')
            print(results)
            self.log('valid_ap', 0)
        else:
            input_dict = {"y_true": y_true, "y_pred": y_pred}
            try:
                self.log('valid_' + self.metric, self.evaluator.eval(input_dict)[self.metric], sync_dist=True)
            except:
                pass

    def test_step(self, batched_data, batch_idx):
        if self.dataset_name in ['PCQM4M-LSC', 'ZINC']:
            y_pred = self(batched_data).view(-1)
            y_true = batched_data.y.view(-1)
        else:
            y_pred = self(batched_data)
            y_true = batched_data.y
        return {
            'y_pred': y_pred,
            'y_true': y_true
        }

    def test_epoch_end(self, outputs):
        y_pred = torch.cat([i['y_pred'] for i in outputs])
        y_true = torch.cat([i['y_true'] for i in outputs])
        if self.dataset_name == 'role_prediction':
            take_to_results_test, split_result = eval_role_prediction_perf(self.trainer.datamodule.dataset['test_dataset'], self)
            for take, results in take_to_results_test.items():
                self.log(f'{take}_micro_f1', results['micro_f1'])
                self.log(f'{take}_macro_f1', results['macro_f1'])
            print(f'Result For Split:')
            print(split_result)

        if self.dataset_name == 'ogbg-molpcba':
            mask = ~torch.isnan(y_true)
            loss = self.loss_fn(y_pred[mask], y_true[mask])
            self.log('test_ap', loss, sync_dist=True)
        elif self.dataset_name == 'role_prediction':
            y_pred = y_pred.argmax(1).cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
            results = classification_report(y_true, y_pred, labels=list(range(len(self.label_names))),
                                            target_names=self.label_names, output_dict=True)

            self.log('test_micro_f1', results['accuracy'] if 'accuracy' in results else results['micro avg']['f1-score'])
            self.log('test_macro_f1', results['macro avg']['f1-score'])

            results = classification_report(y_true, y_pred, labels=list(range(len(self.label_names))),
                                            target_names=self.label_names)

            print('Test Results:')
            print(results)
            self.log('test_ap', 0)

            output_role_predictions(self, train_dataset=self.trainer.datamodule.dataset['train_dataset'],
                                    val_dataset=self.trainer.datamodule.dataset['valid_dataset'],
                                    test_dataset=self.trainer.datamodule.dataset['test_dataset'],
                                    save_name=self.role_prediction_save_name)
        else:
            input_dict = {"y_true": y_true, "y_pred": y_pred}
            try:
                self.log('test_' + self.metric, self.evaluator.eval(input_dict)[self.metric], sync_dist=True)
            except:
                pass

    def configure_optimizers(self):
        if self.dataset_name == 'role_prediction':
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=0.001, weight_decay=0.001)
        return [optimizer]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Graphormer")
        parser.add_argument('--n_layers', type=int, default=12)
        parser.add_argument('--num_heads', type=int, default=32)
        parser.add_argument('--hidden_dim', type=int, default=512)
        parser.add_argument('--ffn_dim', type=int, default=512)
        parser.add_argument('--intput_dropout_rate', type=float, default=0.1)
        parser.add_argument('--dropout_rate', type=float, default=0.1)
        parser.add_argument('--weight_decay', type=float, default=0.01)
        parser.add_argument('--attention_dropout_rate',
                            type=float, default=0.1)
        parser.add_argument('--checkpoint_path', type=str, default='')
        parser.add_argument('--warmup_updates', type=int, default=60000)
        parser.add_argument('--tot_updates', type=int, default=1000000)
        parser.add_argument('--peak_lr', type=float, default=2e-4)
        parser.add_argument('--end_lr', type=float, default=1e-9)
        parser.add_argument('--edge_type', type=str, default='multi_hop')
        parser.add_argument('--validate', action='store_true', default=False)
        parser.add_argument('--test', action='store_true', default=False)
        parser.add_argument('--flag', action='store_true')
        parser.add_argument('--flag_m', type=int, default=3)
        parser.add_argument('--flag_step_size', type=float, default=1e-3)
        parser.add_argument('--flag_mag', type=float, default=1e-3)
        parser.add_argument('--role_prediction_save_name', type=str, default=None)
        return parent_parser


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
