import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
import numpy as np

# Load data
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

# Set index
state_df = state_df.set_index('Date')

# Train-test split
train = state_df.iloc[:-8]
test = state_df.iloc[-8:]

# SARIMA model
model = SARIMAX(
    train['Total'],
    order=(1,1,1),
    seasonal_order=(1,1,1,12)
)

model_fit = model.fit()

# Forecast
predictions = model_fit.forecast(steps=8)

# Evaluation
mae = mean_absolute_error(
    test['Total'],
    predictions
)

sarima_rmse = np.sqrt(
    ((test['Total'] - predictions) ** 2).mean()
)

print("MAE:", mae)
print("SARIMA RMSE:", sarima_rmse)