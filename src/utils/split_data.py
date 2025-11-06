import os
import shutil
import random
import json

# --- Configuration ---
# Get the directory where this script (split_data.py) is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up two levels (from src/utils to taco-sort-mlops/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Define paths relative to the project root
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PREPARED_DIR = os.path.join(PROJECT_ROOT, 'data', 'prepared')

# Path to the COCO annotations file
ANNOTATIONS_FILE = os.path.join(RAW_DIR, 'annotations.json')

# Path where the intermediate YOLO .txt labels will be saved
RAW_LABELS_DIR = os.path.join(RAW_DIR, 'labels')

VAL_RATIO = 0.2
# ---------------------

print("Starting dataset preparation...")
print(f"Project root set to: {PROJECT_ROOT}")

# ---
# === PART 1: Convert COCO (annotations.json) to YOLO (.txt files) ===
# ---
print("\n--- Part 1: Converting COCO to YOLO format ---")

# Create the output directory for YOLO labels
os.makedirs(RAW_LABELS_DIR, exist_ok=True)
print(f"Created/Ensured raw labels directory at: {RAW_LABELS_DIR}")

# Load the COCO JSON file
try:
    with open(ANNOTATIONS_FILE, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: Annotations file not found at {ANNOTATIONS_FILE}")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {ANNOTATIONS_FILE}")
    exit()

# Create lookup maps for faster processing
images_map = {img['id']: img for img in data['images']}
annotations_map = {}
if 'annotations' in data:
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_map:
            annotations_map[img_id] = []
        annotations_map[img_id].append(ann)

print(f"Loaded {len(images_map)} image entries and {len(data.get('annotations', []))} annotation entries.")

# Function to convert COCO bbox to YOLO format
def coco_to_yolo(coco_bbox, img_w, img_h):
    # coco_bbox = [x_min, y_min, w, h]
    x_min, y_min, w, h = coco_bbox
    
    # Calculate center coordinates
    x_center = x_min + w / 2
    y_center = y_min + h / 2
    
    # Normalize coordinates
    x_center_norm = x_center / img_w
    y_center_norm = y_center / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    
    return x_center_norm, y_center_norm, w_norm, h_norm

# Process each image and create its .txt label file
converted_count = 0
for img_id, img_data in images_map.items():
    img_w = img_data['width']
    img_h = img_data['height']
    
    # e.g., "batch_1/000006.jpg" -> "000006"
    base_name = os.path.splitext(os.path.basename(img_data['file_name']))[0]
    label_path = os.path.join(RAW_LABELS_DIR, f"{base_name}.txt")
    
    yolo_lines = []
    
    # Check if this image has any annotations
    if img_id in annotations_map:
        for ann in annotations_map[img_id]:
            class_id = ann['category_id']
            coco_bbox = ann['bbox']
            
            # Convert
            x_c, y_c, w, h = coco_to_yolo(coco_bbox, img_w, img_h)
            
            # Format as string
            yolo_lines.append(f"{class_id} {x_c} {y_c} {w} {h}\n")
    
    # Write the .txt file (even if it's empty, for "no object" images)
    with open(label_path, 'w') as f:
        f.writelines(yolo_lines)
    
    if yolo_lines:
        converted_count += 1

print(f"Conversion complete. Created {len(images_map)} .txt label files in {RAW_LABELS_DIR}.")
print(f"({converted_count} files contain annotations, {len(images_map) - converted_count} are empty).")


# ---
# === PART 2: Split images and new .txt labels into train/val sets ===
# ---
print("\n--- Part 2: Splitting dataset into train/val ---")

# Get the list of images directly from the JSON data we already loaded
images_list = list(images_map.values()) # List of image dictionaries

# Shuffle and split
random.seed(42)  # for reproducible splits
random.shuffle(images_list)
split_idx = int(len(images_list) * (1 - VAL_RATIO))
train_images = images_list[:split_idx]
val_images = images_list[split_idx:]

print(f"Splitting into {len(train_images)} train and {len(val_images)} val images.")

# 3. Create directories and copy files
for subset, images_in_subset in [('train', train_images), ('val', val_images)]:
    img_subset_dir = os.path.join(PREPARED_DIR, 'images', subset)
    lbl_subset_dir = os.path.join(PREPARED_DIR, 'labels', subset)
    
    os.makedirs(img_subset_dir, exist_ok=True)
    os.makedirs(lbl_subset_dir, exist_ok=True)
    
    print(f"Processing {subset} set...")
    copied_labels = 0
    
    for img_data in images_in_subset:
        # Get paths and names from the img_data dictionary
        img_file_path_relative = img_data['file_name'] # e.g., "batch_1/000006.jpg"
        img_file_name = os.path.basename(img_file_path_relative) # e.g., "000006.jpg"
        base_name = os.path.splitext(img_file_name)[0] # e.g., "000006"
        label_file_name = f"{base_name}.txt"
        
        # --- Define source and destination paths ---
        
        # Source image path
        src_img_path = os.path.join(RAW_DIR, img_file_path_relative) 
        
        # Source label path (from the directory we just created in Part 1)
        src_label_path = os.path.join(RAW_LABELS_DIR, label_file_name) 
        
        # Destination paths
        dst_img_path = os.path.join(img_subset_dir, img_file_name)
        dst_label_path = os.path.join(lbl_subset_dir, label_file_name)

        # Copy image
        if not os.path.exists(src_img_path):
             print(f"Warning: Source image not found: {src_img_path}")
             continue
        shutil.copy(src_img_path, dst_img_path)
        
        # Check if label exists and copy it
        if os.path.exists(src_label_path):
            shutil.copy(src_label_path, dst_label_path)
            copied_labels += 1
        else:
            # This should not happen if Part 1 ran correctly
            print(f"Warning: Label file not found for {img_file_name} at {src_label_path}")

    print(f"Copied {len(images_in_subset)} images and {copied_labels} labels to {subset} set.")

print("---")
print("Dataset preparation complete.")
print(f"Train/Val splits are ready in {PREPARED_DIR}")