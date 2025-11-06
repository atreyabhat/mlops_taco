import mlflow
import mlflow.onnx  
import onnx         
from ultralytics import YOLO
from pathlib import Path

# Connect to MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("taco_sort_training")

# Define parameters
params = {
    "model_type": "./taco-sort-mlops/yolo11m.pt",
    "epochs": 50,
    "imgsz": 640,
    "lr0": 0.01
}

# Start a new run
with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"Starting run: {run_id}")
    mlflow.log_params(params)
    
    # Train model
    model = YOLO(params["model_type"])
    results = model.train(
        data="taco.yaml", 
        epochs=params["epochs"], 
        imgsz=params["imgsz"], 
        lr0=params["lr0"],
        device="mps",
        workers=16,
        batch=16
    )

    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    best_model = YOLO(best_model_path)

    # Export the model to ONNX format. 
    onnx_path = best_model.export(format='onnx')
    print(f"Successfully exported to: {onnx_path}")

    onnx_model = onnx.load(onnx_path)

    # Log the loaded ONNX model object to MLflow
    print("Logging and registering ONNX model...")
    mlflow.onnx.log_model(
        onnx_model=onnx_model,  
        artifact_path="model", 
        registered_model_name="taco_sort_yolo",
    )

    # Log other artifacts
    mlflow.log_artifact(str(Path(results.save_dir) / "results.png"))
    mlflow.log_artifact(str(Path(results.save_dir) / "confusion_matrix.png"))

