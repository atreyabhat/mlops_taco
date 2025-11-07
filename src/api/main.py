import mlflow
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from PIL import Image
import io
import numpy as np
import os
import json
import datetime
import pandas as pd
from contextlib import asynccontextmanager
import traceback
from prometheus_fastapi_instrumentator import Instrumentator  # 6.1: Import Prometheus

# --- Globals ---
model = None
PREDICTION_LOG_FILE = "data/logs/prediction_logs.jsonl"  # 6.2: Log file


# --- Pydantic Models ---
class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str


class PredictionResponse(BaseModel):
    boxes: list[BBox]


# --- Lifespan Event ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    # 6.2: Create logs directory on startup
    os.makedirs(os.path.dirname(PREDICTION_LOG_FILE), exist_ok=True)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    print(f"Setting MLflow tracking URI to: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)

    model_uri = "models:/taco_sort_yolo@staging"
    print(f"Loading model from: {model_uri}")

    model = mlflow.pyfunc.load_model(model_uri)
    print("Model loaded successfully from Registry.")

    yield

    print("Cleaning up and shutting down...")
    model = None


# --- App Creation ---
app = FastAPI(title="TACO-SORT API", lifespan=lifespan)

# 6.1: Add Prometheus metrics endpoint (e.g., /metrics)
Instrumentator().instrument(app).expose(app)


# --- API Endpoints ---
@app.get("/health")
def health_check():
    if model:
        return {"status": "ok", "model_loaded": True}
    return {"status": "error", "model_loaded": False}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="Confidence threshold"),
    iou: float = Query(0.7, ge=0.0, le=1.0, description="IoU threshold for NMS"),
):
    try:
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes))

        input_df = pd.DataFrame([img], columns=["image"])
        runtime_params = {"conf": conf, "iou": iou}

        results_df = model.predict(input_df, params=runtime_params)

        # 6.2: Log proxy metrics for this prediction
        try:
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "num_boxes_predicted": len(results_df),
                "avg_confidence": np.nan_to_num(results_df["confidence"].mean()),
                "class_distribution": results_df["name"].value_counts().to_dict(),
            }
            with open(PREDICTION_LOG_FILE, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as log_e:
            print(f"WARNING: Failed to log prediction: {log_e}")

        # Convert to API response
        response_boxes = []
        for _, row in results_df.iterrows():
            response_boxes.append(
                BBox(
                    x1=row["xmin"],
                    y1=row["ymin"],
                    x2=row["xmax"],
                    y2=row["ymax"],
                    confidence=row["confidence"],
                    class_name=row["name"],
                )
            )

        return {"boxes": response_boxes}

    except Exception as e:
        print("--- PREDICTION FAILED ---")
        print(traceback.format_exc())
        print("-------------------------")
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error: {str(e)}. Check server logs for full traceback.",
        )


if __name__ == "__main__":
    print("Running in local debug mode...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
