import mlflow
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from PIL import Image
import io
import numpy as np
import os 
from contextlib import asynccontextmanager
import traceback 
import pandas as pd # The model input needs to be a DataFrame

# Define the API response format using Pydantic
class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str

class PredictionResponse(BaseModel):
    boxes: list[BBox]

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    print(f"Setting MLflow tracking URI to: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Load the model with the '@staging' alias
    model_uri = "models:/taco_sort_yolo@staging"
    
    model = mlflow.pyfunc.load_model(model_uri)
    print("Model loaded successfully from Registry.")

    yield

    print("Cleaning up and shutting down...")
    model = None

app = FastAPI(title="TACO-SORT API", lifespan=lifespan)

@app.get("/health")
def health_check():
    if model:
        return {"status": "ok", "model_loaded": True}
    return {"status": "error", "model_loaded": False}

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(...),
    # Add query parameters for runtime settings
    conf: float = Query(0.25, ge=0.0, le=1.0, description="Confidence threshold"),
    iou: float = Query(0.7, ge=0.0, le=1.0, description="IoU threshold for NMS")
):
    try:
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes))
        
        # The 'pyfunc' wrapper expects a pandas DataFrame
        input_df = pd.DataFrame([img], columns=["image"])
        
        # Bundle query params into a dictionary
        runtime_params = {"conf": conf, "iou": iou}
    
        results_df = model.predict(input_df, params=runtime_params) 
        
        # Convert DataFrame to our Pydantic response model
        response_boxes = []
        for _, row in results_df.iterrows():
            response_boxes.append(
                BBox(
                    x1=row['xmin'],
                    y1=row['ymin'],
                    x2=row['xmax'],
                    y2=row['ymax'],
                    confidence=row['confidence'],
                    class_name=row['name']
                )
            )
            
        return {"boxes": response_boxes}
    
    except Exception as e:
        print("--- PREDICTION FAILED ---")
        print(traceback.format_exc())
        print("-------------------------")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal Server Error: {str(e)}. Check server logs for full traceback."
        )

if __name__ == "__main__":
    print("Running in local debug mode...")
    uvicorn.run(app, host="0.0.0.0", port=8000)