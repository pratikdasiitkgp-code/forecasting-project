import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

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

# Date conversion
df['Date'] = pd.to_datetime(
    df['Date'],
    format='mixed',
    dayfirst=True
)

# Select one state
state_df = df[df['State'] == 'Alabama']

# Group by date
state_df = (
    state_df.groupby('Date')['Total']
    .sum()
    .reset_index()
)

# Sort by date
state_df = state_df.sort_values('Date')

# Only sales values
data = state_df['Total'].values

# Scaling
scaler = MinMaxScaler()

data_scaled = scaler.fit_transform(
    data.reshape(-1,1)
)

# Create sequences
X = []
y = []

sequence_length = 8

for i in range(sequence_length, len(data_scaled)):

    X.append(
        data_scaled[i-sequence_length:i]
    )

    y.append(
        data_scaled[i]
    )

X = np.array(X)
y = np.array(y)

# Train-test split
split_index = int(len(X) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

# LSTM Model
model = Sequential()

model.add(
    LSTM(
        64,
        activation='relu',
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    )
)

model.add(Dense(1))

# Compile
model.compile(
    optimizer='adam',
    loss='mse'
)

# Train
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=8,
    validation_data=(X_test, y_test)
)

# Predictions
predictions = model.predict(X_test)

# Inverse scaling
predictions = scaler.inverse_transform(
    predictions
)

y_test_actual = scaler.inverse_transform(
    y_test
)

# Evaluation
mae = mean_absolute_error(
    y_test_actual,
    predictions
)

lstm_rmse = np.sqrt(
    np.mean(
        (y_test_actual - predictions) ** 2
    )
)

print("MAE:", mae)
print("LSTM RMSE:", lstm_rmse)

# Plot
plt.figure(figsize=(10,5))

plt.plot(
    y_test_actual,
    label='Actual'
)

plt.plot(
    predictions,
    label='Predicted'
)

plt.legend()

plt.title("LSTM Forecast")

plt.show()