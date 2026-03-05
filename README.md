# Yolo-World for UAS based Object Detection

## Overview

The goal of this project was to fine-tune the YOLO World open vocabulary object detection model (https://github.com/AILab-CVC/YOLO-World) with the VisDrone dataset 
(https://github.com/VisDrone/VisDrone-Dataset) in order to achieve better, real-time performance on a UAS system. This process required converting the VisDrone
dataset to the appropriate format including a data.yaml file, running a conversion script the converted the annotations from the default custom `.txt` format
to the YOLO PyTorch format, and setting up the corresponding YOLO-World config files. 

## 📖 Table of Contents
* [Setup](#Setup)
* [Installation & Setup](#installation--setup)
* [Results](RESULTS.md)

# Setup

## Activate the enviornment

```
conda activate yoloworld_stable
```
## Converting the dataset

The VisDrone dataset must be converted into the standard YOLO format, pictured below
```
datasets/VisDrone/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

The format for data.yaml is
```
path: /mnt/andre/YOLO-World/datasets/VisDrone
train: images/train
val: images/val
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
```

## Example command to run inference with config/weights on a .mp4 file 
```
# Ensure you are in the YOLO-World root directory
PYTHONPATH=. python demo/video_demo.py \
    configs/pretrain/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py \
    weights/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth \
    VID_01.mp4 \
    "person, backpack, soccer ball, landing pad" \
    --score-thr 0.05 \
    --out video_outputs/VID_01_detected.mp4 \
    --device cuda:0
```


