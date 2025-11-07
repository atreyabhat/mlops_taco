import mlflow
import pandas as pd
import numpy as np
import glob
import os
import json
import time
from PIL import Image

# --- Config ---
MLFLOW_URI = "http://127.0.0.1:5001"
MODEL_URI = "models:/taco_sort_yolo@production"
VAL_IMAGES_PATH = "data/prepared/images/val/"
OUTPUT_LOG_PATH = "data/reference_data.jsonl" # Use .jsonl for logs

def main():
    print(f"Loading reference model from: {MODEL_URI}")
    mlflow.set_tracking_uri(MLFLOW_URI)
    model = mlflow.pyfunc.load_model(MODEL_URI)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_LOG_PATH), exist_ok=True)
    
    # Find all validation images
    image_files = glob.glob(os.path.join(VAL_IMAGES_PATH, "*.jpg"))
    print(f"Found {len(image_files)} validation images.")
    
    with open(OUTPUT_LOG_PATH, "w") as f:
        for img_path in image_files:
            try:
                img = Image.open(img_path)
                
                # Create the DataFrame input for the pyfunc model
                input_df = pd.DataFrame([img], columns=["image"])
                
                # Make prediction
                df = model.predict(input_df)
                
                # --- THIS IS THE FIX ---
                # We must explicitly convert all numpy types to native Python types
                # for JSON serialization.
                
                num_boxes = int(len(df)) # Cast numpy.int64 to int
                
                if num_boxes > 0:
                    # FIX 1: Use .item() to convert numpy.float32 to python float
                    avg_confidence = np.mean(df['confidence'].values).item()
                    
                    # FIX 2: Explicitly cast values in the dict to python int
                    class_counts = df['name'].value_counts()
                    class_distribution = {str(k): int(v) for k, v in class_counts.items()}
                else:
                    avg_confidence = 0.0
                    class_distribution = {}

                # Create the log entry
                entry = {
                    "timestamp": time.time(),
                    "num_boxes_predicted": num_boxes,
                    "avg_confidence": avg_confidence,
                    "class_distribution": class_distribution
                }
                
                # Write the log line
                f.write(json.dumps(entry) + "\n")
                
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")
                
    print(f"Saving reference data to {OUTPUT_LOG_PATH}...")

if __name__ == "__main__":
    main()