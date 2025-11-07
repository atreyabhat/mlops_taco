import mlflow
import mlflow.pyfunc
from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import shutil
import os


# --- Custom MLflow Wrapper ---
class UltralyticsWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        model_path = context.artifacts["model_path"]
        self.model = YOLO(model_path)
        print("Custom UltralyticsWrapper loaded successfully.")

    def predict(self, context, model_input, params=None):
        conf = 0.25  # Default confidence threshold
        iou = 0.7  # Default IoU threshold for NMS

        if params:
            conf = params.get("conf", 0.25)
            iou = params.get("iou", 0.7)

        # Iterate through each row (each image)
        all_boxes = []
        for i, row in model_input.iterrows():
            img = row[0]  # Get the PIL Image from the first column

            results = self.model.predict(img, conf=conf, iou=iou)

            # Convert the results to the DataFrame API expects
            boxes = results[0].boxes.data
            df = pd.DataFrame(
                boxes.cpu().numpy(),
                columns=["xmin", "ymin", "xmax", "ymax", "confidence", "class_id"],
            )

            df["name"] = df["class_id"].apply(lambda x: self.model.names[int(x)])

            # We only need the final DataFrame
            return df[["xmin", "ymin", "xmax", "ymax", "confidence", "name"]]


# Connect to MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("taco_sort_training")

# Define training parameters
params = {
    "model_type": "./taco-sort-mlops/yolo11m.pt",
    "epochs": 100,
    "imgsz": 640,
    "lr0": 0.01,
}

# Start a new MLflow run
with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"Starting run: {run_id}")
    mlflow.log_params(params)

    model = YOLO(params["model_type"])

    # Train the model
    results = model.train(
        data="taco.yaml",
        epochs=params["epochs"],
        imgsz=params["imgsz"],
        lr0=params["lr0"],
        device="mps",
        workers=16,
        batch=16,
    )

    # Get the path to the best .pt file
    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Best model saved to: {best_model_path}")

    # Log other artifacts (charts)
    mlflow.log_artifact(str(Path(results.save_dir) / "results.png"))
    mlflow.log_artifact(str(Path(results.save_dir) / "confusion_matrix.png"))

    # Define the artifacts for the MLflow model
    artifacts = {"model_path": str(best_model_path)}

    # Log the model using our custom wrapper
    print("Logging model with custom pyfunc wrapper...")
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=UltralyticsWrapper(),
        artifacts=artifacts,
        registered_model_name="taco_sort_yolo",
    )

    print(f"Run ID: {run_id} - Training complete. Custom model logged and registered.")
