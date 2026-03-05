from ultralytics import YOLOWorld

# Baseline Medium Model Weights
#model = YOLOWorld('yolov8m-worldv2.pt')

# Baseline X-Large Model Weights
#model = YOLOWorld('yolov8x-worldv2.pt')

# Fine-Tuned for 5 epochs while freezing CLIP encoder
model = YOLOWorld('runs/detect/runs/VisDrone_Research/yolo_world_preserved/weights/best.pt')

# Fine-Tuned for 40 epochs while freezing CLIP encoder
#model = YOLOWorld('runs/detect/runs/VisDrone_NewPath/yolo_world_balanced/weights/best.pt')

# 2. Define classes to detect
#target_classes = ["backpack", "soccer ball", "orange landing pad", "pedestrian", "car"]

# Descriptive Classes
target_classes = [
    "orange circle landing pad",        
    "backpack", 
    "pedestrian",                                   # VisDrone categroy
    "car",                                          # VisDrone category
    "soccer ball",      
    "foreign object on driveway",                    # general prompt for anything not explicitly prompted (crocs, cooler)
    "foreign object on orange circle landing pad"   # general prompt for anything on top of the landing pad
]
model.set_classes(target_classes)

# 3. Run inference
results = model.predict(
    source='VID_01.mp4', 
    imgsz=1280,      
    conf=0.10,       
    save=True,       # Automatically saves plotted video to runs/detect/predict/
    device=0,        
    project='runs/inference',
    name='drone_test_native'
)

print(f"Inference complete. Output saved to runs/inference/drone_test_native/")
