import torch
from ultralytics import YOLOWorld

# Path MUST match exactly where the error occurred
save_path = '/mnt/andre/YOLO-World/datasets/VisDrone/images/text_embeddings_clip_ViT-B_32.pt'
class_names = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]

print("--- Generating Dictionary-style Embeddings on CPU ---")
model = YOLOWorld('yolov8m-world.pt')
model.to('cpu')

# model.get_text_pe returns a tensor of [1, 10, 512]
txt_feats = model.model.get_text_pe(class_names, batch=len(class_names))

# IMPORTANT: Convert to Dictionary { "class_name": tensor }
# We squeeze(0) to remove the batch dimension
txt_map = {name: feat for name, feat in zip(class_names, txt_feats.squeeze(0))}

torch.save(txt_map, save_path)
print(f"Success! Saved dictionary cache to {save_path}")
