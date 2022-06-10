import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG

from scene_graph_prediction.pointnet2_dir.pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG


# class PointNet2ClassificationMSG(PointNet2ClassificationSSG): # original
#     def _build_model(self):
#         super()._build_model()
#
#         self.SA_modules = nn.ModuleList()
#         self.SA_modules.append(
#             PointnetSAModuleMSG(
#                 npoint=512,
#                 radii=[0.1, 0.2, 0.4],
#                 nsamples=[16, 32, 128],
#                 mlps=[[self.input_dim-3, 32, 32, 64], [self.input_dim-3, 64, 64, 128], [self.input_dim-3, 64, 96, 128]],
#                 use_xyz=True,
#             )
#         )
#
#         input_channels = 64 + 128 + 128
#         self.SA_modules.append(
#             PointnetSAModuleMSG(
#                 npoint=128,
#                 radii=[0.2, 0.4, 0.8],
#                 nsamples=[32, 64, 128],
#                 mlps=[
#                     [input_channels, 64, 64, 128],
#                     [input_channels, 128, 128, 256],
#                     [input_channels, 128, 128, 256],
#                 ],
#                 use_xyz=True,
#             )
#         )
#         self.SA_modules.append(
#             PointnetSAModule(
#                 mlp=[128 + 256 + 256, 256, 512, 1024],
#                 use_xyz=True,
#             )
#         )
class PointNet2ClassificationMSG(PointNet2ClassificationSSG):  # max we can run
    def _build_model(self):
        super()._build_model()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.1, 0.2],
                nsamples=[16, 32],
                mlps=[[self.input_dim - 3, 64, 64], [self.input_dim - 3, 64, 128]],
                use_xyz=True,
            )
        )

        input_channels = 64 + 128
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4],
                nsamples=[32, 64],
                mlps=[
                    [input_channels, 128, 128],
                    [input_channels, 128, 128],
                ],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[128 + 128, 256, 256],
                use_xyz=True,
            )
        )

# class PointNet2ClassificationMSG(PointNet2ClassificationSSG):  # light
#     def _build_model(self):
#         super()._build_model()
#
#         self.SA_modules = nn.ModuleList()
#         self.SA_modules.append(
#             PointnetSAModuleMSG(
#                 npoint=512,
#                 radii=[0.1, 0.2],
#                 nsamples=[16, 32],
#                 mlps=[[self.input_dim - 3, 32, 32], [self.input_dim - 3, 32, 32]],
#                 use_xyz=True,
#             )
#         )
#
#         input_channels = 32 + 32
#         self.SA_modules.append(
#             PointnetSAModuleMSG(
#                 npoint=128,
#                 radii=[0.2, 0.4],
#                 nsamples=[32, 64],
#                 mlps=[
#                     [input_channels, 64, 64],
#                     [input_channels, 64, 64],
#                 ],
#                 use_xyz=True,
#             )
#         )
#         self.SA_modules.append(
#             PointnetSAModule(
#                 mlp=[64 + 64, 256, 256],
#                 use_xyz=True,
#             )
#         )


# class PointNet2ClassificationMSG(PointNet2ClassificationSSG):  # Extremely light
#     def _build_model(self):
#         super()._build_model()
#         self.SA_modules = nn.ModuleList()
#         self.SA_modules.append(
#             PointnetSAModule(
#                 npoint=512,
#                 radius=0.2,
#                 nsample=64,
#                 mlp=[self.input_dim - 3, 32],
#                 use_xyz=True,
#             )
#         )
#         self.SA_modules.append(
#             PointnetSAModule(
#                 npoint=128,
#                 radius=0.4,
#                 nsample=64,
#                 mlp=[32, 32],
#                 use_xyz=True,
#             )
#         )
#         self.SA_modules.append(
#             PointnetSAModule(
#                 mlp=[32, 256], use_xyz=True
#             )
#         )
