import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import numpy as np

# Load dataset
df = pd.read_csv(
    '../data/Forecasting Case- Study Sheet1.csv'
)

# Preprocessing
df['Total'] = (
    df['Total']
    .str.replace(',', '')
    .astype(float)
)

df['Date'] = pd.to_datetime(df['Date'],format='mixed',dayfirst=True)

# Example state
state_df = df[df['State'] == 'Alabama']

# Group by date
state_df = (
    state_df.groupby('Date')['Total']
    .sum()
    .reset_index()
)

# Prophet format
state_df.columns = ['ds', 'y']

# Train-test split
train = state_df.iloc[:-8]

test = state_df.iloc[-8:]

# Model
model = Prophet()

model.fit(train)

# Future dataframe
future = model.make_future_dataframe(
    periods=8,
    freq='W'
)

forecast = model.predict(future)

# Predictions
predictions = forecast['yhat'].tail(8).values

# Actual values
actual = test['y'].values

# Metrics
mae = mean_absolute_error(
    actual,
    predictions
)

prophet_rmse = np.sqrt(
    ((actual - predictions) ** 2).mean()
)

print("MAE:", mae)
print("Prophet RMSE:", prophet_rmse)

# Forecast plot
fig = model.plot(forecast)