import cv2
import os
import random
import yaml
from pathlib import Path

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PREPARED_DIR = PROJECT_ROOT / "data" / "prepared"
IMAGE_DIR = PREPARED_DIR / "images" / "val"
LABEL_DIR = PREPARED_DIR / "labels" / "val"
YAML_FILE = PROJECT_ROOT / "taco.yaml"
OUTPUT_DIR = PROJECT_ROOT / "data" / "verification_output"
NUM_IMAGES_TO_CHECK = 5
# ---------------------


def yolo_to_pixel_bbox(yolo_bbox, img_w, img_h):
    """
    Converts a normalized YOLO format [x_center, y_center, w, h]
    to pixel coordinates [x_min, y_min, x_max, y_max].
    """
    x_center_norm, y_center_norm, w_norm, h_norm = yolo_bbox

    # De-normalize
    x_center_pix = x_center_norm * img_w
    y_center_pix = y_center_norm * img_h
    w_pix = w_norm * img_w
    h_pix = h_norm * img_h

    # Calculate pixel coordinates
    x_min = int(x_center_pix - (w_pix / 2))
    y_min = int(y_center_pix - (h_pix / 2))
    x_max = int(x_center_pix + (w_pix / 2))
    y_max = int(y_center_pix + (h_pix / 2))

    # Clamp values to image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_w - 1, x_max)
    y_max = min(img_h - 1, y_max)

    return x_min, y_min, x_max, y_max


def main():
    print("Starting label verification...")

    # 1. Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output will be saved to: {OUTPUT_DIR}")

    # 2. Load class names from taco.yaml
    try:
        with open(YAML_FILE, "r") as f:
            class_names = yaml.safe_load(f)["names"]
        print(f"Loaded {len(class_names)} class names from {YAML_FILE.name}")
    except Exception as e:
        print(f"Error: Could not read {YAML_FILE}. {e}")
        return

    # 3. Get list of all validation images
    image_files = list(IMAGE_DIR.glob("*.jpg"))
    if not image_files:
        print(f"Error: No images found in {IMAGE_DIR}")
        return

    # 4. Pick random images
    selected_images = random.sample(
        image_files, min(NUM_IMAGES_TO_CHECK, len(image_files))
    )
    print(f"Randomly selected {len(selected_images)} images to verify...")

    # 5. Process each selected image
    for img_path in selected_images:
        print(f"\nProcessing: {img_path.name}")

        # Load the image
        img = cv2.imread(str(img_path))
        if img is None:
            print("  - Error: Could not read image.")
            continue

        img_h, img_w, _ = img.shape

        # Find the corresponding label file
        label_file_name = f"{img_path.stem}.txt"
        label_path = LABEL_DIR / label_file_name

        if not label_path.exists():
            print("  - Warning: No label file found. Saving blank image.")
        else:
            with open(label_path, "r") as f:
                lines = f.readlines()

            if not lines:
                print("  - Info: Label file is empty (no objects).")

            # Draw each bounding box
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"  - Warning: Skipping malformed line: {line.strip()}")
                    continue

                try:
                    class_id = int(parts[0])
                    yolo_bbox = [float(p) for p in parts[1:]]

                    # Convert coords
                    x_min, y_min, x_max, y_max = yolo_to_pixel_bbox(
                        yolo_bbox, img_w, img_h
                    )

                    # Get class name
                    class_name = class_names.get(class_id, f"ID:{class_id}")

                    # Draw box and label
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(
                        img,
                        class_name,
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                    )

                    print(
                        f"  - Found: {class_name} at [{x_min}, {y_min}, {x_max}, {y_max}]"
                    )

                except Exception as e:
                    print(f"  - Error processing line '{line.strip()}': {e}")

        # 6. Save the new image
        output_path = OUTPUT_DIR / f"VERIFIED_{img_path.name}"
        cv2.imwrite(str(output_path), img)
        print(f"  - Saved to: {output_path.name}")

    print("\nVerification complete. Check the 'data/verification_output' folder.")


if __name__ == "__main__":
    main()
