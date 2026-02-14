import os
from PIL import Image

def convert_yolo_to_dainet_format(image_folder, label_folder, output_txt_path):
    """
    Converts YOLO format (class x_center y_center w h - normalized) 
    to DAI-Net format (path num_objs x y w h c - absolute).
    """
    with open(output_txt_path, 'w') as out_f:
        # Iterate over all files in the image folder
        for filename in os.listdir(image_folder):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            image_path = os.path.join(image_folder, filename)
            label_file = filename.rsplit('.', 1)[0] + '.txt'
            label_path = os.path.join(label_folder, label_file)
            
            # If no label file exists, skip or treat as 0 objects
            if not os.path.exists(label_path):
                continue
                
            # Get image dimensions to convert normalized coordinates to absolute
            try:
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                print(f"Error opening image {image_path}: {e}")
                continue

            annotations = []
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    # YOLO format: class_id x_center y_center w h (all normalized 0-1)
                    # We ignore parts[0] (class_id) if we assume all are people (class 1)
                    
                    x_c, y_c, w_norm, h_norm = map(float, parts[1:5])
                    
                    # Convert to Absolute Top-Left (x, y, w, h)
                    w_abs = w_norm * img_width
                    h_abs = h_norm * img_height
                    x_abs = (x_c * img_width) - (w_abs / 2)
                    y_abs = (y_c * img_height) - (h_abs / 2)
                    
                    # Class c should be 1 for objects (0 is background in SSD/DSFD)
                    c = 1 
                    
                    annotations.extend([x_abs, y_abs, w_abs, h_abs, c])

            # Construct the line: path num_objs [x y w h c]...
            num_objs = len(lines)
            if num_objs > 0:
                ann_str = " ".join(map(str, annotations))
                # Ensure image_path is valid relative to where you run train.py
                out_f.write(f"{image_path} {num_objs} {ann_str}\n")

# Usage Example:
# Adjust these paths to where you extracted your Roboflow data
# convert_yolo_to_dainet_format('dataset/train/images', 'dataset/train/labels', 'dataset/train_people.txt')
# convert_yolo_to_dainet_format('dataset/valid/images', 'dataset/valid/labels', 'dataset/val_people.txt')