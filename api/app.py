from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load(
    '../models/xgboost_model.pkl'
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