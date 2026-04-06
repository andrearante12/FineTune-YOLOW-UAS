# Yolo-World for UAS based Object Detection

## Overview

The goal of this project was to fine-tune the YOLO World open vocabulary object detection model (https://github.com/AILab-CVC/YOLO-World) with the VisDrone dataset 
(https://github.com/VisDrone/VisDrone-Dataset) in order to achieve better, real-time performance on a UAS system. This process required converting the VisDrone
dataset to the appropriate format including a data.yaml file, running a conversion script the converted the annotations from the default custom `.txt` format
to the COCO dataset format, and setting up the corresponding YOLO-World config files. 

## 📖 Table of Contents
* [Setup](#Setup)
* [Enviorment Setup for Finetuning](./docs/INSTALLATION-FINETUNE.md)
* [Results](./docs/RESULTS.md)
* [Evaluation Guide](./docs/EVALUATION.md)
* [Project Scripts](#project-scripts)

## Requirements

The model was finetuned on a system with the following specs:
* **GPU:** 2x NVIDIA RTX 5090 (32GB)
* **CPU:** AMD Threadripper PRO 7955WX (16C/32T)
* **RAM:** 512GB DDR5
* **OS:** Ubuntu 22.04 LTS
* **Software:** CUDA 12.8, PyTorch 2.12.0 

## Setup



* [Enviornment Setup for Finetuning](./docs/INSTALLATION-FINETUNE.md)


## Inference

Run from the root dir to perform inference.
```
python scripts/run_inference.py
```

To modify the target video adjust this section in ./scripts/run_inference.py. This example runs on VID_01.mp4 and detects target classes defined below.

```
# Descriptive Classes
target_classes = [
    "orange circle landing pad",        
    "backpack", 
    "pedestrian",                                  
    "car",                                          
    "soccer ball",      
    "foreign object on driveway",                    
    "foreign object on orange circle landing pad"  
]
model.set_classes(target_classes)

results = model.predict(
    source='VID_01.mp4', 
    imgsz=1280,      
    conf=0.10,       
    save=True,       # Automatically saves plotted video to runs/detect/predict/
    device=0,        
    project='runs/inference',
    name='drone_test_native'
)
```

## Project Scripts

All custom scripts are in the `scripts/` directory. Run from the repo root.

| Script | Purpose |
|--------|---------|
| `scripts/finetune.py` | Fine-tune YOLOWorld on VisDrone with frozen CLIP encoder |
| `scripts/run_inference.py` | Video inference with custom descriptive class prompts |
| `scripts/yolo_world_jetson.py` | Real-time inference on NVIDIA Jetson |
| `scripts/evaluate.py` | Closed-vocab VisDrone evaluation (COCO mAP) |
| `scripts/evaluate_open_vocab.py` | Open-vocab evaluation on novel classes |
| `scripts/sweep_checkpoints.py` | Evaluate multiple checkpoints across hyperparameters |
| `scripts/convert_visdrone.py` | Convert VisDrone annotations to YOLO format |
| `scripts/yolo_to_coco.py` | Convert YOLO labels to COCO JSON |
| `scripts/verify_coco_format.py` | Validate COCO annotation format |
| `scripts/extract_frames.py` | Extract frames from video for dataset creation |
| `scripts/fix_embeddings.py` | Generate text embeddings for class names |
| `scripts/test_frozen_weights.py` | Inspect checkpoint metadata |
