import mlflow
import mlflow.pyfunc
from ultralytics import YOLO
from pathlib import Path
import pandas as pd


# --- Custom MLflow Wrapper ---
class UltralyticsWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        model_path = context.artifacts["model_path"]
        self.model = YOLO(model_path)
        print("Custom UltralyticsWrapper loaded successfully.")

    def predict(self, context, model_input, params=None):
        conf = 0.25
        iou = 0.7

        if params:
            conf = params.get("conf", 0.25)
            iou = params.get("iou", 0.7)

        all_boxes = []
        for i, row in model_input.iterrows():
            img = row.iloc[0]
            results = self.model.predict(img, conf=conf, iou=iou)
            boxes = results[0].boxes.data
            df = pd.DataFrame(
                boxes.cpu().numpy(),
                columns=["xmin", "ymin", "xmax", "ymax", "confidence", "class_id"],
            )
            df["name"] = df["class_id"].apply(lambda x: self.model.names[int(x)])
            return df[["xmin", "ymin", "xmax", "ymax", "confidence", "name"]]


# Connect to MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("taco_sort_training")

# Define training parameters with augmentation
params = {
    "model_type": "./taco-sort-mlops/yolo11m.pt",
    "epochs": 50,
    "imgsz": 640,  # larger to help small objects
    "lr0": 0.001,  # lower LR for fine-tuning
    "batch": 16,  # maybe smaller batch if high res
    # Augmentation parameters
    "hsv_h": 0.01,
    "hsv_s": 0.5,
    "hsv_v": 0.3,
    "degrees": 5.0,
    "translate": 0.05,
    "scale": 0.8,
    "shear": 2.0,
    "perspective": 0.001,
    "fliplr": 0.5,
    "flipud": 0.0,
}

# Start MLflow run
with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"Starting run: {run_id}")
    mlflow.log_params(params)

    model = YOLO(params["model_type"])

    # Train with augmentation parameters
    results = model.train(
        data="taco.yaml",
        epochs=params["epochs"],
        imgsz=params["imgsz"],
        lr0=params["lr0"],
        device="mps",
        workers=16,
        batch=16,
        # Add augmentation parameters
        # hsv_h=params["hsv_h"],
        # hsv_s=params["hsv_s"],
        # hsv_v=params["hsv_v"],
        # degrees=params["degrees"],
        # translate=params["translate"],
        # scale=params["scale"],
        # shear=params["shear"],
        # perspective=params["perspective"],
        # fliplr=params["fliplr"],
        # flipud=params["flipud"],
    )

    # Log artifacts
    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    mlflow.log_artifact(str(Path(results.save_dir) / "results.png"))
    mlflow.log_artifact(str(Path(results.save_dir) / "confusion_matrix.png"))

    # Log model
    artifacts = {"model_path": str(best_model_path)}
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=UltralyticsWrapper(),
        artifacts=artifacts,
        registered_model_name="taco_sort_yolo",
    )

    print(f"Run ID: {run_id} - Training complete with augmentations.")
