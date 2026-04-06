import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from ultralytics import YOLOWorld

def main():
    # 1. Load the YOLOWorld (medium) checkpoint
    model = YOLOWorld('yolov8m-worldv2.pt')

    # 2. Freeze the text encoder (the "Language Brain")
    for name, param in model.model.named_parameters():
        if "txt_model" in name:
            param.requires_grad = False
            print(f"Frozen: {name}")

    # 3. Define target training classes (VisDrone labels)
    classes = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
    model.set_classes(classes)

    # 4. Launch the Optimized Training
    model.train(
        data='datasets/VisDrone/data.yaml',
        epochs=80,
        imgsz=1280,
        batch=24,
        lr0=2e-4,
        weight_decay=0.05,
        warmup_epochs=3,
        amp=True,
        device=[0, 1],
        save_period=5,
        project='runs/VisDrone_NewPath',
        name='yolo_world_80epoch'
    )

if __name__ == '__main__':
    main()
