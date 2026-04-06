import os
from PIL import Image
from tqdm import tqdm

def convert_visdrone_to_yolo(img_dir, ann_dir, out_dir):
    if not os.path.exists(ann_dir):
        print(f"Error: Annotation directory not found: {ann_dir}")
        return 0
    
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    ann_files = [f for f in os.listdir(ann_dir) if f.endswith('.txt')]
    
    # Path Debugging
    if ann_files:
        test_ann = ann_files[0]
        test_img = os.path.join(img_dir, test_ann.replace('.txt', '.jpg'))
        print(f"DEBUG: Checking file: {test_ann}")
        print(f"DEBUG: Looking for image at: {test_img}")
        print(f"DEBUG: Image exists? {os.path.exists(test_img)}")

    for ann_file in tqdm(ann_files):
        img_path = os.path.join(img_dir, ann_file.replace('.txt', '.jpg'))
        
        if not os.path.exists(img_path):
            continue

        try:
            with Image.open(img_path) as img:
                w, h = img.size
            
            with open(os.path.join(ann_dir, ann_file), 'r') as f:
                yolo_lines = []
                for line in f:
                    line = line.strip()
                    if not line: continue
                    
                    # Handle both comma-separated and space-separated formats
                    parts = line.split(',') if ',' in line else line.split()
                    
                    if len(parts) < 6:
                        continue
                    
                    # VisDrone format: <left>, <top>, <width>, <height>, <score>, <category>
                    try:
                        cls_id = int(parts[5])
                        
                        # Filter for standard VisDrone classes (1-10)
                        # 0 and 11 are usually 'ignored' or 'others'
                        if cls_id < 1 or cls_id > 10:
                            continue
                        
                        # Map 1-10 to 0-9 for YOLO
                        target_cls = cls_id - 1
                        
                        # Extract BBox (Pixel Coordinates)
                        left, top, bw, bh = map(float, parts[:4])
                        
                        # Calculate YOLO format (Normalized 0.0 to 1.0)
                        x_center = (left + bw / 2) / w
                        y_center = (top + bh / 2) / h
                        norm_bw = bw / w
                        norm_bh = bh / h
                        
                        # Clamp values between 0 and 1 to prevent out-of-bounds errors
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        norm_bw = max(0, min(1, norm_bw))
                        norm_bh = max(0, min(1, norm_bh))
                        
                        yolo_lines.append(f"{target_cls} {x_center:.6f} {y_center:.6f} {norm_bw:.6f} {norm_bh:.6f}")
                    except (ValueError, IndexError):
                        continue
                
            if yolo_lines:
                with open(os.path.join(out_dir, ann_file), 'w') as f:
                    f.write('\n'.join(yolo_lines))
                count += 1
                
        except Exception as e:
            print(f"Error processing {ann_file}: {e}")
            
    return count

# Using Absolute Paths for the 5090 Workstation environment
base_path = '/mnt/andre/YOLO-World/datasets/VisDrone'

for split in ['train', 'val']:
    print(f"\n--- Processing {split} set ---")
    img_path = os.path.abspath(f'{base_path}/images/{split}')
    ann_path = os.path.abspath(f'{base_path}/annotations/{split}')
    out_path = os.path.abspath(f'{base_path}/labels/{split}')
    
    num = convert_visdrone_to_yolo(img_path, ann_path, out_path)
    print(f"Result: Successfully converted {num} files.")
