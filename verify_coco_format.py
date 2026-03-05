import cv2
import os
from ultralytics.utils.plotting import Annotator

# --- UPDATE THESE PATHS ---
img_name = "9999982_00000_d_0000053.jpg" # Replace with a real filename from your images/train folder
img_path = f"/mnt/andre/YOLO-World/datasets/VisDrone/images/train/{img_name}"
label_path = f"/mnt/andre/YOLO-World/datasets/VisDrone/labels/train/{img_name.replace('.jpg', '.txt')}"
# --------------------------

if not os.path.exists(img_path):
    print(f"Error: Could not find image at {img_path}")
    exit()

img = cv2.imread(img_path)
h, w, _ = img.shape
annotator = Annotator(img)

with open(label_path, 'r') as f:
    for line in f.readlines():
        cls, x, y, nw, nh = map(float, line.split())
        # Convert YOLO normalized to pixel coordinates
        x1 = (x - nw/2) * w
        y1 = (y - nh/2) * h
        x2 = (x + nw/2) * w
        y2 = (y + nh/2) * h
        annotator.box_label([x1, y1, x2, y2], label=f"Class {int(cls)}")

# Save the result so you can view it, since you are on a server/CLI
output_path = "verification_result.jpg"
cv2.imwrite(output_path, annotator.result())
print(f"Done! Check {output_path} to see the boxes.")
