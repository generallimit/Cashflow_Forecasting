from fastapi import FastAPI
import numpy as np

app = FastAPI()

@app.get("/predict_cash_flow/")
def predict_cash_flow(features: str):
    features_array = np.array([float(x) for x in features.split(",")]).reshape(1, -1)
    prediction = model_xgb.predict(features_array)[0]
    return {"predicted_cash_flow": float(prediction)}
