# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from role_prediction.graphormer.collator import collator
from role_prediction.graphormer.wrapper import MyRolePredictionDataset
from pytorch_lightning import LightningDataModule
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from functools import partial
from collections import defaultdict

dataset = None


def get_dataset(dataset_name='abaaba'):
    global dataset
    if dataset is not None:
        return dataset

    # max_node is set to max(max(num_val_graph_nodes), max(num_test_graph_nodes))
    if dataset_name == 'role_prediction':
        dataset = {
            'num_class': 5,
            'metric': 'ap',
            'metric_mode': 'max',
            'evaluator': None,  # same objective function, so reuse it
            'train_dataset': MyRolePredictionDataset(split='train'),
            'valid_dataset': MyRolePredictionDataset(split='val'),
            'test_dataset': MyRolePredictionDataset(split='val'),
            # TODO you can also select test here, especially if you have access to either GT or predictions from previous steps.
            'max_node': 64,
        }
    else:
        raise NotImplementedError

    print(f' > {dataset_name} loaded!')
    print(dataset)
    print(f' > dataset info ends')
    return dataset


class GraphDataModule(LightningDataModule):
    name = "OGB-GRAPH"

    def __init__(
            self,
            dataset_name: str = 'ogbg-molpcba',
            num_workers: int = 0,
            batch_size: int = 256,
            seed: int = 42,
            multi_hop_max_dist: int = 5,
            spatial_pos_max: int = 1024,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.dataset = get_dataset(self.dataset_name)

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset_train = ...
        self.dataset_val = ...
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max

    def setup(self, stage: str = None):
        if self.dataset_name == 'role_prediction':
            self.dataset_train = self.dataset['train_dataset']
            self.dataset_val = self.dataset['valid_dataset']
            try:
                self.dataset_test = self.dataset['test_dataset']
            except KeyError:
                print('Not creating test dataset')

        else:
            split_idx = self.dataset['dataset'].get_idx_split()
            self.dataset_train = self.dataset['dataset'][split_idx["train"]]
            self.dataset_val = self.dataset['dataset'][split_idx["valid"]]
            self.dataset_test = self.dataset['dataset'][split_idx["test"]]

    def compute_sample_weights(self, dataset):
        count = defaultdict(int)
        for elem in dataset:
            count[elem[1].item()] += 1

        weights_per_class = [0.] * len(count)
        for cls, count in sorted(count.items()):
            weights_per_class[cls] = 1 / float(count)

        weight = [0] * len(dataset)
        for idx, elem in enumerate(dataset):
            class_idx = elem[1].item()
            weight[idx] = weights_per_class[class_idx]
        return weight

    def train_dataloader(self):
        if 'role_prediction' in self.dataset_name:
            weight = self.compute_sample_weights(self.dataset_train)
            sampler = torch.utils.data.WeightedRandomSampler(weights=weight, num_samples=len(weight))
            shuffle = False
        else:
            sampler = None
            shuffle = True

        loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=partial(collator, max_node=get_dataset(self.dataset_name)[
                'max_node'], multi_hop_max_dist=self.multi_hop_max_dist, spatial_pos_max=self.spatial_pos_max, dataset_name=self.dataset_name),
        )
        print('len(train_dataloader)', len(loader))
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=partial(collator, max_node=get_dataset(self.dataset_name)[
                'max_node'], multi_hop_max_dist=self.multi_hop_max_dist, spatial_pos_max=self.spatial_pos_max, dataset_name=self.dataset_name),
        )
        print('len(val_dataloader)', len(loader))
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=partial(collator, max_node=get_dataset(self.dataset_name)[
                'max_node'], multi_hop_max_dist=self.multi_hop_max_dist, spatial_pos_max=self.spatial_pos_max, dataset_name=self.dataset_name),
        )
        print('len(test_dataloader)', len(loader))
        return loader
