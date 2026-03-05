import os
import json
import cv2

def yolo_to_coco(img_dir, label_dir, output_json, classes):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(classes)]
    }
    
    ann_id = 0
    for img_id, img_name in enumerate(os.listdir(img_dir)):
        if not img_name.endswith(('.jpg', '.png', '.jpeg')): continue
        
        # Load image for dimensions
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        
        coco["images"].append({
            "id": img_id, "file_name": img_name, "width": width, "height": height
        })
        
        label_path = os.path.join(label_dir, img_name.rsplit('.', 1)[0] + '.txt')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    cls, x_c, y_c, w, h = map(float, line.split())
                    # Convert YOLO (normalized) to COCO (absolute pixel x, y, w, h)
                    abs_w = w * width
                    abs_h = h * height
                    abs_x = (x_c * width) - (abs_w / 2)
                    abs_y = (y_c * height) - (abs_h / 2)
                    
                    coco["annotations"].append({
                        "id": ann_id, "image_id": img_id, "category_id": int(cls),
                        "bbox": [abs_x, abs_y, abs_w, abs_h],
                        "area": abs_w * abs_h, "iscrowd": 0
                    })
                    ann_id += 1
                    
    with open(output_json, 'w') as f:
        json.dump(coco, f)
    print(f"Successfully created {output_json}")

# Configuration
vis_classes = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
yolo_to_coco('datasets/VisDrone/images/train', 'datasets/VisDrone/labels/train', 'datasets/VisDrone/annotations/train.json', vis_classes)
yolo_to_coco('datasets/VisDrone/images/val', 'datasets/VisDrone/labels/val', 'datasets/VisDrone/annotations/val.json', vis_classes)
