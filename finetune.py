import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from ultralytics import YOLOWorld

def main():
    # 1. Load the original pre-trained model for a fresh start
    model = YOLOWorld('yolov8m-worldv2.pt')

    # 2. HARD GUARD: Freeze the text encoder (the "Language Brain")
    # This prevents the model from "forgetting" what a soccer ball is
    for name, param in model.model.named_parameters():
        if "txt_model" in name:
            param.requires_grad = False
            print(f"✅ Frozen: {name}")

    # 3. Define your target training classes (VisDrone labels)
    classes = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
    model.set_classes(classes)

    # 4. Launch the Optimized Training
    model.train(
        data='datasets/VisDrone/data.yaml',
        epochs=40,               # 40 is the "sweet spot" for fine-tuning stability
        imgsz=1280,              # Critical for spotting small UAS objects
        batch=24,                # Safe limit for your 32GB VRAM buffer
        lr0=2e-4,                # Official recommended lower LR for preservation
        weight_decay=0.05,       # High anchor to prevent Catastrophic Forgetting
        warmup_epochs=3,         # Gentle start to protect pre-trained weights
        amp=True,                
        device=[0, 1],           # Use both 5090s
        project='runs/VisDrone_NewPath',
        name='yolo_world_balanced'
    )

if __name__ == '__main__':
    main()
