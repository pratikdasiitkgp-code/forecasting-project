from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Model path
MODEL_PATH = BASE_DIR / "models" / "xgboost_model.pkl"

# Frontend path
FRONTEND_PATH = BASE_DIR / "frontend" / "index.html"


# Load trained model
model = joblib.load(MODEL_PATH)


app = FastAPI(
    title="Sales Forecasting API",
    version="1.0.0"
)


# Input schema
class PredictionInput(BaseModel):

    lag_1: float
    lag_7: float
    lag_30: float

    rolling_mean_7: float
    rolling_std_7: float

    month: int
    week: int
    quarter: int

    is_holiday: int


# Home API
@app.get("/")
def home():

    return {
        "message": "Forecast API Running"
    }


# Prediction webpage
@app.get("/predict-ui")
def prediction_page():

    return FileResponse(FRONTEND_PATH)


# Prediction API
@app.post("/predict")
def predict(data: PredictionInput):

    input_data = pd.DataFrame([{

        "lag_1": data.lag_1,
        "lag_7": data.lag_7,
        "lag_30": data.lag_30,

        "rolling_mean_7": data.rolling_mean_7,
        "rolling_std_7": data.rolling_std_7,

        "month": data.month,
        "week": data.week,
        "quarter": data.quarter,

        "is_holiday": data.is_holiday

    }])


    prediction = model.predict(input_data)[0]


    return {
        "prediction": float(prediction)
    }