from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(
    BASE_DIR / "models" / "xgboost_model.pkl"
)

@app.get("/")
def home():
    return {
        "message": "Forecast API Running"
    }

@app.post("/predict")
def predict(data: dict):

    features = pd.DataFrame([data])

    prediction = model.predict(features)

    return {
        "prediction": prediction.tolist()
    }