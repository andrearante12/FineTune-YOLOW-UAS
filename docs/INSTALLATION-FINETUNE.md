# Setup

## Download the conda enviornment

```
conda env create -f yoloworld_stable.yml
```

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

## Setting up the YOLOWorld Config files for Visdrone

`configs/visdrone/yolo_world_v2_x_visdrone.py`

```
dataset_type = 'MultiModalDataset'
data_root = 'data/VisDrone/'

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        dataset=dict(
            type='YOLOv5CocoDataset',
            data_root=data_root,
            ann_file='annotations/train_coco.json',
            data_prefix=dict(img='train/images/')),
        class_text_path='data/texts/visdrone_texts.json',
        pipeline=train_pipeline))

```