## HigherHRNet (2D Human Pose Detection)

Original Code: https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation  
Adaptations: We crea te a new 4D-OR dataset, with 14 joints. For this we change some configurations. Most of the code is the same as the original. We add a new
script to infer the 2D poses and output it to files.

- Recommended PyTorch Version: pytorch==1.10.0
- conda create -n HigherHRNet python=3.7
- conda activate HigherHRNet
- conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
- `cd` into HigherHrNet and run `pip install -r requirements.txt`
- Run `wget https://github.com/egeozsoy/4D-OR/releases/download/v0.1/HigherHRNet_files.zip` and unzip
- Move `pose_higher_hrnet_w32_512.pth` to `models/pytorch/pose_coco/pose_higher_hrnet_w32_512.pth`
- (Optional) To use the pretrained model, move `OR_4D_kpt` to `output/OR_4D_kpt`
- (Optional) Run `python -m tools.dist_train --cfg config.yaml` to train 2D pose detection. You can instead use the pretrained weights as well.
- (Optional) Run `python -m tools.calculate_2d_human_pose --cfg config.yaml` to output 2D pose predictions. This only outputs for
  the test dataset, so make sure to set the test dataset to train/val/test etc. to output for all data. Alternatively, you can directly use the outputs
  pred_or_4d*.npz downloaded from the 4D-OR github repo.

## VoxelPose (3D Human Pose Detection)

Original Code: https://github.com/microsoft/voxelpose-pytorch  
Adaptations: We create a new 4D-OR dataset, with 14 joints. For this we change some configurations. Most of the code is the same as the original. We add a new
script to infer the 3D poses and output it to files.

- Recommended PyTorch Version: pytorch==1.4.0
- conda create -n VoxelPose python=3.7
- conda activate VoxelPose
- conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
- `cd` into voxelpose and run `pip install -r requirements.txt`
- Run `wget https://github.com/egeozsoy/4D-OR/releases/download/v0.1/VoxelPose_files.zip` and unzip
- (Optional) To use the pretrained model, Move `OR_4D` to `output/OR_4D`
- (Optional) Run `python -m run.train_3d --cfg configs/4d_or/config.yaml` to train 3D pose detection. You can instead use the pretrained weights as well.
- (Optional) Run `python get_human_poses_4D_OR.py --cfg configs/4d_or/config.yaml` to output 3D pose predictions. Set test dataset to train/val/test to output
  for
  whole data. Alternatively, you can directly use the 3D pose outputs in by putting them in the data folder. (Move `OR_4D_outputs` to `data/OR_4D_outputs`)

## GroupFree3D (3D Object Pose Detection)

Original Code: https://github.com/zeliu98/Group-Free-3D  
Adaptations: We create a new 4D-OR dataset. For this we change some configurations. Most of the code is the same as the original. We add a new script to infer
the 3D poses and output it to files.

- Recommended PyTorch Version: pytorch==1.10.0
- conda create -n GroupFree3D python=3.7
- conda activate GroupFree3D
- conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
- `cd` into group_free_3D and run `pip install -r requirements.txt`
- Run `wget https://github.com/egeozsoy/4D-OR/releases/download/v0.1/GroupFree3D_files.zip` and unzip
- (Optional) To use the pretrained model, Move `ckpt_epoch_last.pth` to `pretrained_model/ckpt_epoch_last.pth`
- Run `CUDA_HOME=/usr/local/cuda-11.3 sh init.sh`
- `cd` out -> `cd ../..` (to root dir)
- (Optional) Run `python -m external_src.group_free_3D.train_dist` to train 3D object detection. You can instead use the pretrained weights as well. In both
  cases, put the relevant checkpoint into the pretrained_model folder before the next step.
- (Optional) Run `python -m external_src.group_free_3D.infer` to generate predictions. Set DATASET_SPLIT to train,val and test one after another. Alternatively,
  you can directly use the 3D object outputs in by putting them in the GROUP_FREE_PREDICTIONS_PATH folder. (Download from github `group_free_predictions`)
