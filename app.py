from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import joblib
import pandas as pd
from typing import List

app = FastAPI(title="Cash Flow Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the model and scaler
try:
    model = joblib.load("model_xgb.joblib")
    scaler = joblib.load("scaler.joblib")
except FileNotFoundError:
    print("Warning: Model files not found. Please train the model first.")
    model = None
    scaler = None

class PredictionInput(BaseModel):
    features: List[float]

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.post("/predict")
async def predict_cash_flow(input_data: PredictionInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")
    
    try:
        # Convert input to numpy array and reshape
        features = np.array(input_data.features).reshape(1, -1)
        
        # Scale the features
        scaled_features = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        
        return {
            "predicted_cash_flow": float(prediction),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 