'''
The code here is modified from https://github.com/charlesq34/pointnet under MIT License
'''
if __name__ == '__main__' and __package__ is None:
    from os import sys

    sys.path.append('../')

import torch.nn as nn
from scene_graph_prediction.pointnet2_dir.pointnet2.models.pointnet2_msg_cls import PointNet2ClassificationMSG


class PointNetfeat(nn.Module):
    def __init__(self, input_dim=6, out_size=1024, input_dropout=0.0):
        super(PointNetfeat, self).__init__()
        self.name = 'pnetenc'
        self.backbone = PointNet2ClassificationMSG(input_dim=input_dim)
        self.out_size = out_size
        self.input_dropout = input_dropout

    def forward(self, x):
        assert x.ndim > 2
        x = x.transpose(1, 2)
        x = self.backbone(x, return_features=True)[:, :, 0]
        return x
