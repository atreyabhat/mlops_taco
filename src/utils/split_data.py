import os
import shutil
import random
import json

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PREPARED_DIR = os.path.join(PROJECT_ROOT, "data", "prepared")
ANNOTATIONS_FILE = os.path.join(RAW_DIR, "annotations.json")
RAW_LABELS_DIR = os.path.join(RAW_DIR, "labels")

VAL_RATIO = 0.2

print("Starting dataset preparation...")
print(f"Project root set to: {PROJECT_ROOT}")

# Part 1: Convert COCO annotations to YOLO format
print("\n--- Part 1: Converting COCO to YOLO format ---")

os.makedirs(RAW_LABELS_DIR, exist_ok=True)
print(f"Created/Ensured raw labels directory at: {RAW_LABELS_DIR}")

try:
    with open(ANNOTATIONS_FILE, "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: Annotations file not found at {ANNOTATIONS_FILE}")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {ANNOTATIONS_FILE}")
    exit()

# Create lookup maps for faster processing
images_map = {img["id"]: img for img in data["images"]}
annotations_map = {}
if "annotations" in data:
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations_map:
            annotations_map[img_id] = []
        annotations_map[img_id].append(ann)

print(
    f"Loaded {len(images_map)} image entries and {len(data.get('annotations', []))} annotation entries."
)


def coco_to_yolo(coco_bbox, img_w, img_h):
    """Convert COCO bbox [x_min, y_min, w, h] to YOLO format [x_center, y_center, w, h] normalized"""
    x_min, y_min, w, h = coco_bbox

    x_center = x_min + w / 2
    y_center = y_min + h / 2

    x_center_norm = x_center / img_w
    y_center_norm = y_center / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    return x_center_norm, y_center_norm, w_norm, h_norm


# Process each image and create its .txt label file
converted_count = 0
files_created = 0
for img_id, img_data in images_map.items():
    img_w = img_data["width"]
    img_h = img_data["height"]

    # --- FIX: Use os.path.splitext to handle .jpg, .JPG, .png ---
    # e.g., "batch_1/000008.jpg" -> "batch_1_000008"
    # e.g., "batch_7/000044.JPG" -> "batch_7_000044"
    temp_img_name = img_data["file_name"].replace("/", "_")
    unique_base_name = os.path.splitext(temp_img_name)[0]
    # --- End of Fix ---

    label_file_name = f"{unique_base_name}.txt"
    label_path = os.path.join(RAW_LABELS_DIR, label_file_name)

    yolo_lines = []

    if img_id in annotations_map:
        for ann in annotations_map[img_id]:
            class_id = ann["category_id"]
            coco_bbox = ann["bbox"]

            x_c, y_c, w, h = coco_to_yolo(coco_bbox, img_w, img_h)
            yolo_lines.append(f"{class_id} {x_c} {y_c} {w} {h}\n")

    with open(label_path, "w") as f:
        f.writelines(yolo_lines)
    files_created += 1

    if yolo_lines:
        converted_count += 1

print(
    f"Conversion complete. Created {files_created} .txt label files in {RAW_LABELS_DIR}."
)
print(
    f"({converted_count} files contain annotations, {files_created - converted_count} are empty)."
)

# Part 2: Split images and labels into train/val sets
print("\n--- Part 2: Splitting dataset into train/val ---")

images_list = list(images_map.values())

random.seed(42)
random.shuffle(images_list)
split_idx = int(len(images_list) * (1 - VAL_RATIO))
train_images = images_list[:split_idx]
val_images = images_list[split_idx:]

print(f"Splitting into {len(train_images)} train and {len(val_images)} val images.")

for subset, images_in_subset in [("train", train_images), ("val", val_images)]:
    img_subset_dir = os.path.join(PREPARED_DIR, "images", subset)
    lbl_subset_dir = os.path.join(PREPARED_DIR, "labels", subset)

    os.makedirs(img_subset_dir, exist_ok=True)
    os.makedirs(lbl_subset_dir, exist_ok=True)

    print(f"Processing {subset} set...")
    copied_images = 0
    copied_labels = 0

    for img_data in images_in_subset:
        img_file_path_relative = img_data["file_name"]  # "batch_1/000008.jpg"

        # --- FIX: Use the same unique naming scheme as Part 1 ---
        # Handle mixed extensions like .JPG, .jpg, .png
        temp_img_name = img_file_path_relative.replace("/", "_")
        base_name = os.path.splitext(temp_img_name)[0]

        # Standardize on .jpg for all output images
        unique_img_name = f"{base_name}.jpg"
        label_file_name = f"{base_name}.txt"
        # --- End of Fix ---

        src_img_path = os.path.join(RAW_DIR, img_file_path_relative)
        src_label_path = os.path.join(RAW_LABELS_DIR, label_file_name)
        dst_img_path = os.path.join(img_subset_dir, unique_img_name)
        dst_label_path = os.path.join(lbl_subset_dir, label_file_name)

        if not os.path.exists(src_img_path):
            print(f"Warning: Source image not found: {src_img_path}")
            continue

        shutil.copy(src_img_path, dst_img_path)
        copied_images += 1

        if os.path.exists(src_label_path):
            # Only copy the label if it's not empty
            if os.path.getsize(src_label_path) > 0:
                shutil.copy(src_label_path, dst_label_path)
                copied_labels += 1
            else:
                # If the image had no annotations, we still create an empty .txt file
                # This is what YOLO expects.
                open(dst_label_path, "w").close()
        else:
            print(
                f"Warning: Label file not found for {img_file_path_relative} at {src_label_path}"
            )
            # Create an empty file anyway, as the image exists
            open(dst_label_path, "w").close()

    print(
        f"Copied {copied_images} images and {copied_labels} non-empty labels to {subset} set."
    )

print("---")
print("Dataset preparation complete.")
print(f"Train/Val splits are ready in {PREPARED_DIR}")
