import pandas as pd
import numpy as np
import joblib
import holidays

from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------
# Project paths
# ---------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "Forecasting Case- Study Sheet1.csv"
MODEL_PATH = BASE_DIR / "models" / "xgboost_model.pkl"


# ---------------------------------------
# 1. Load dataset
# ---------------------------------------
df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df["Total"] = df["Total"].str.replace(",", "")
df["Total"] = df["Total"].astype(float)

df = df.sort_values(["State", "Date"])

df = df.rename(columns={
    "Total": "Sales"
})


# ---------------------------------------
# 2. Select Alabama data
# ---------------------------------------
state_df = df[df["State"] == "Alabama"].copy()


# ---------------------------------------
# 3. Aggregate weekly sales
# ---------------------------------------
state_df = (
    state_df.groupby("Date")["Sales"]
    .sum()
    .reset_index()
)

state_df = state_df.set_index("Date")

all_dates = pd.date_range(
    start=state_df.index.min(),
    end=state_df.index.max(),
    freq="W"
)

state_df = state_df.reindex(all_dates)

state_df["Sales"] = state_df["Sales"].interpolate()

state_df = state_df.reset_index()

state_df = state_df.rename(columns={
    "index": "Date"
})


# ---------------------------------------
# 4. Feature Engineering
# ---------------------------------------

# Lag features
state_df["lag_1"] = state_df["Sales"].shift(1)
state_df["lag_7"] = state_df["Sales"].shift(7)
state_df["lag_30"] = state_df["Sales"].shift(30)

# Rolling features
state_df["rolling_mean_7"] = (
    state_df["Sales"]
    .rolling(window=7)
    .mean()
)

state_df["rolling_std_7"] = (
    state_df["Sales"]
    .rolling(window=7)
    .std()
)

# Calendar features
state_df["month"] = state_df["Date"].dt.month

state_df["week"] = (
    state_df["Date"]
    .dt.isocalendar()
    .week
    .astype(int)
)

state_df["quarter"] = state_df["Date"].dt.quarter


# Indian holiday feature
india_holidays = holidays.India()

state_df["is_holiday"] = state_df["Date"].apply(
    lambda x: 1 if x in india_holidays else 0
)


# Remove missing values created by lag/rolling features
state_df = state_df.dropna()


# ---------------------------------------
# 5. Features used by XGBoost
# ---------------------------------------
features = [
    "lag_1",
    "lag_7",
    "lag_30",
    "rolling_mean_7",
    "rolling_std_7",
    "month",
    "week",
    "quarter",
    "is_holiday"
]


# ---------------------------------------
# 6. Train/Test split
# Last 8 weeks = test data
# ---------------------------------------
train = state_df.iloc[:-8]
test = state_df.iloc[-8:]

X_train = train[features]
y_train = train["Sales"]

X_test = test[features]
y_test = test["Sales"]


# ---------------------------------------
# 7. XGBoost Model
# ---------------------------------------
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5
)

model.fit(X_train, y_train)


# ---------------------------------------
# 8. Predictions
# ---------------------------------------
predictions = model.predict(X_test)


# ---------------------------------------
# 9. Evaluation
# ---------------------------------------
mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")


# ---------------------------------------
# 10. Save trained model
# ---------------------------------------
MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_PATH
)

print(f"Model saved to: {MODEL_PATH}")