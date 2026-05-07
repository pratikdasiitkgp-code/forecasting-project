# End-to-End Time Series Forecasting System with API

## Project Overview

This project is a production-style time series forecasting system developed for forecasting next 8 weeks of sales for different states using historical sales data.

The system:
- trains multiple forecasting models
- compares model performance
- automatically selects the best model
- exposes predictions through a REST API using FastAPI

---

# Problem Statement

Forecast the next 8 weeks of sales for each state using historical data.

The system handles:
- missing dates
- missing values
- seasonality
- trends
- feature engineering
- automatic model selection

---

# Models Implemented

The following forecasting models were implemented and compared:

1. XGBoost
2. SARIMA
3. Facebook Prophet
4. LSTM

---

# Feature Engineering

The following features were created:

- Lag Features
  - lag_1
  - lag_7
  - lag_30

- Rolling Statistics
  - rolling_mean_7
  - rolling_std_7

- Date Features
  - month
  - week
  - quarter

- Holiday Feature
  - is_holiday

---

# Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Statsmodels
- Prophet
- TensorFlow / Keras
- FastAPI
- Uvicorn

---

# Project Structure

```text
forecasting_project/
│
├── data/
├── notebooks/
├── src/
├── api/
├── models/
├── requirements.txt
└── README.md
```

---

# Dataset

Dataset contains:
- State
- Date
- Total Sales
- Category

Dataset preprocessing included:
- date conversion
- missing value handling
- sales value conversion
- sorting
- aggregation

---

# Train-Test Split

Time-series split was used to avoid data leakage.

- Training Data → historical data
- Testing Data → last 8 weeks

---

# Evaluation Metrics

Models were evaluated using:

- MAE
- RMSE

Lower RMSE indicates better forecasting performance.

---

# Best Model Selection

The best model was automatically selected based on minimum RMSE.

Example:

```python
best_model = min(results, key=results.get)
```

---

# API Development

FastAPI was used to create a REST API.

API Endpoints:

## Home Endpoint

```bash
GET /
```

Returns:

```json
{
  "message": "Forecast API Running"
}
```

---

## Prediction Endpoint

```bash
POST /predict
```

Example Input:

```json
{
  "lag_1": 1000000,
  "lag_7": 1200000,
  "lag_30": 1100000,
  "rolling_mean_7": 1150000,
  "rolling_std_7": 50000,
  "month": 5,
  "week": 18,
  "quarter": 2,
  "is_holiday": 0
}
```

Example Output:

```json
{
  "prediction": [123196088]
}
```

---

# Swagger Documentation

Swagger UI available at:

```bash
http://127.0.0.1:8000/docs
```

---

# Installation

Clone repository:

```bash
git clone <repository_link>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run FastAPI server:

```bash
uvicorn app:app --reload
```

---

# Results

All forecasting models were trained and compared.

The best model was selected automatically based on RMSE.

---

# Future Improvements

- Multi-state parallel forecasting
- Cloud deployment
- Real-time forecasting dashboard
- Automated retraining pipeline
- Docker deployment

---

# Author

Pratik Das