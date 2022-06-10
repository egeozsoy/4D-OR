'''
The code here is modified from https://github.com/charlesq34/pointnet under MIT License
'''
if __name__ == '__main__' and __package__ is None:
    from os import sys

    sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from scene_graph_prediction.scene_graph_helpers.model.pointnets.networks_base import BaseNetwork


class STN3d(nn.Module):
    def __init__(self, point_size=3):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(point_size, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = Variable(torch.from_numpy(
            np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))
        ).view(1, 9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        # iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        iden = Variable(torch.eye(self.k).view(1, self.k * self.k).repeat(batchsize, 1))
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(BaseNetwork):
    def __init__(self, global_feat=True, input_transform=True, feature_transform=False,
                 point_size=3, out_size=1024, batch_norm=True,
                 init_weights=True, pointnet_str: str = None, input_dropout=0.0):
        super(PointNetfeat, self).__init__()
        self.name = 'pnetenc'
        self.use_batch_norm = batch_norm
        self.relu = nn.ReLU()
        self.point_size = point_size
        self.out_size = out_size

        self.conv1 = torch.nn.Conv1d(point_size, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, out_size, 1)
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(out_size)
        self.global_feat = global_feat
        self.input_transform = input_transform
        self.feature_transform = feature_transform
        self.input_dropout = input_dropout

        if input_transform:
            assert pointnet_str is not None
            self.pointnet_str = pointnet_str
            self.stn = STN3d(point_size=point_size)
        if self.feature_transform:
            self.fstn = STNkd(k=64)

        if init_weights:
            self.init_weights('constant', 1, target_op='BatchNorm')
            self.init_weights('xavier_normal', 1)

    def forward(self, x, return_meta=False):
        assert x.ndim > 2
        # Similar to dropout but more efficient as we are actually dropping these points, more like an augmentation
        downsampled_size = int(x.shape[-1] * (1 - self.input_dropout))
        downsample_indices = torch.randperm(x.shape[-1])[:downsampled_size]
        x = x[:, :, downsample_indices]
        n_pts = x.size()[2]
        if self.input_transform:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            if self.pointnet_str is None and self.point_size == 3:
                x[:, :, :3] = torch.bmm(x[:, :, :3], trans)
            elif self.point_size > 3:
                assert self.pointnet_str is not None
                for i in len(self.pointnet_str):
                    p = self.pointnet_str[i]
                    offset = i * 3
                    offset_ = (i + 1) * 3
                    if p == 'p' or p == 'n':  # point and normal
                        x[:, :, offset:offset_] = torch.bmm(x[:, :, offset:offset_], trans)
            x = x.transpose(2, 1)
        else:
            trans = torch.zeros([1])

        x = self.conv1(x)
        if self.use_batch_norm:
            self.bn1(x)
        x = self.relu(x)

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = torch.zeros([1])  # cannot be None in tracing. change to 0
        pointfeat = x
        x = self.conv2(x)
        if self.use_batch_norm:
            self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        if self.use_batch_norm:
            self.bn3(x)
        x = self.relu(x)

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.out_size)

        if self.global_feat:
            if return_meta:
                return x, trans, trans_feat
            else:
                return x

        else:
            x = x.view(-1, self.out_size, 1).repeat(1, 1, n_pts)
            if not return_meta:
                return torch.cat([x, pointfeat], 1)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetCls(BaseNetwork):
    def __init__(self, k=2, in_size=1024, batch_norm=True, drop_out=True, init_weights=True):
        super(PointNetCls, self).__init__()
        self.name = 'pnetcls'
        self.in_size = in_size
        self.k = k
        self.use_batch_norm = batch_norm
        self.use_drop_out = drop_out
        self.fc1 = nn.Linear(in_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        if drop_out:
            self.dropout = nn.Dropout(p=0.3)
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        if init_weights:
            self.init_weights('constant', 1, target_op='BatchNorm')
            self.init_weights('xavier_normal', 1)

    def forward(self, x):
        x = self.fc1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.relu(x)

        x = self.fc2(x)
        if self.use_drop_out:
            x = self.dropout(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.relu(x)

        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class PointNetRelCls(BaseNetwork):

    def __init__(self, k=2, in_size=1024, batch_norm=True, drop_out=True,
                 init_weights=True, image_embedding_size=None, n_object_types=None):
        super(PointNetRelCls, self).__init__()
        self.name = 'pnetcls'
        self.in_size = in_size
        self.use_bn = batch_norm
        self.use_drop_out = drop_out
        self.fc1 = nn.Linear(in_size, 512)
        self.fc2 = nn.Linear(512, 256)
        if image_embedding_size is None:
            image_embedding_size = 0
        self.fc3 = nn.Linear(256 + image_embedding_size + n_object_types * 2, k)
        # self.fc3 = nn.Linear(256 + image_embedding_size, k)
        if self.use_drop_out:
            self.dropout = nn.Dropout(p=0.3)
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        if init_weights:
            self.init_weights('constant', 1, target_op='BatchNorm')
            self.init_weights('xavier_normal', 1)

    def forward(self, x, relation_objects_one_hot=None, image_embeddings=None):
        x = self.fc1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.use_drop_out:
            x = self.dropout(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.relu(x)

        if image_embeddings is not None:  # Late fusion
            image_embeddings = image_embeddings.unsqueeze(0).repeat(len(x), 1)
            x = torch.cat([x, image_embeddings], dim=1)
        if relation_objects_one_hot is not None:  # Also late fusion
            x = torch.cat([x, relation_objects_one_hot], dim=1)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)  # , trans, trans_feat


class PointNetRelClsMulti(BaseNetwork):

    def __init__(self, k=2, in_size=1024, batch_norm=True, drop_out=True,
                 init_weights=True, image_embedding_size=None, n_object_types=None):
        super(PointNetRelClsMulti, self).__init__()
        self.name = 'pnetcls'
        self.in_size = in_size
        self.use_bn = batch_norm
        self.use_drop_out = drop_out
        self.fc1 = nn.Linear(in_size, 512)
        self.fc2 = nn.Linear(512, 256)
        if image_embedding_size is None:
            image_embedding_size = 0
        self.fc3 = nn.Linear(256 + image_embedding_size + n_object_types * 2, k)

        if self.use_drop_out:
            self.dropout = nn.Dropout(p=0.3)
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        if init_weights:
            self.init_weights('constant', 1, target_op='BatchNorm')
            self.init_weights('xavier_normal', 1)

    def forward(self, x, relation_objects_one_hot=None, image_embeddings=None):
        x = self.fc1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.use_drop_out:
            x = self.dropout(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.relu(x)
        if image_embeddings is not None:  # Late fusion
            image_embeddings = image_embeddings.unsqueeze(0).repeat(len(x), 1)
            x = torch.cat([x, image_embeddings], dim=1)
        if relation_objects_one_hot is not None:
            x = torch.cat([x, relation_objects_one_hot], dim=1)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


if __name__ == '__main__':
    model = PointNetCls()
    model.trace('./tmp')
