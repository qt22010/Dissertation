# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 11:32:59 2023

@author: white
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

actual_temperature_data = [15.77, 16.43, 16.55, 19.19, 19.36, 19.17, 17.38]

# Load and preprocess data
data = pd.read_csv(r'C:\Users\white\Desktop\Dissertation\data\weather_data_cleaned1.csv')  # Replace with the actual path
target_temp = 'temp'
y = data[target_temp]

# Normalize the target data
scaler = StandardScaler()
y_scaled = scaler.fit_transform(np.array(y).reshape(-1, 1))

# Create sequences for RNN using all of the available data
sequence_length = len(y_scaled)
sequences = y_scaled

# Build the RNN model
model = Sequential()
model.add(SimpleRNN(units=64, activation='relu', input_shape=(sequence_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(sequences, y_scaled, epochs=50, batch_size=32)

# Make future predictions using the most recent 7 days
input_sequence = sequences[-sequence_length:].reshape(1, -1, 1)

# Create an array to store future predictions
future_predictions = []

# Predict 7 days in the future
for _ in range(7):
    prediction = model.predict(input_sequence)
    future_predictions.append(prediction)
    input_sequence = np.roll(input_sequence, -1, axis=1)
    input_sequence[0, -1, 0] = prediction

# Inverse transform the predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Flatten the future_predictions array
future_predictions_flattened = future_predictions.flatten()

# Calculate MAE and RMSE
mae = mean_absolute_error(actual_temperature_data, future_predictions_flattened)
rmse = mean_squared_error(actual_temperature_data, future_predictions_flattened, squared=False)

print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)

# Create a graph
plt.figure(figsize=(10, 6))
plt.plot(actual_temperature_data[:7], label='Actual Temperatures', marker='o')
plt.plot(future_predictions_flattened[:7], label='Predicted Temperatures', marker='x')

# Draw lines between each actual and predicted point
for i in range(7):
    actual_temp = actual_temperature_data[i]  # Get the actual temperature for the corresponding day
    plt.plot([i, i], [actual_temp, future_predictions_flattened[i]], color='red', linestyle='--')

plt.xlabel('Day')
plt.ylabel('Temperature')
plt.title('Actual vs. Predicted Temperatures')
plt.legend()
plt.grid(True)
plt.show()

###################RAIN

actual_rain_data = [2.21, 0, 11.39, 5.01, 0, 0.26, 10.16]

target_rain = 'rain_1h'
y = data[target_rain]

# Normalize the target data
scaler = StandardScaler()
y_scaled = scaler.fit_transform(np.array(y).reshape(-1, 1))

# Create sequences for RNN using all of the available data
sequence_length = len(y_scaled)
sequences = y_scaled

# Build the RNN model
model = Sequential()
model.add(SimpleRNN(units=64, activation='relu', input_shape=(sequence_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(sequences, y_scaled, epochs=50, batch_size=32)

# Make future predictions using the most recent 7 days
input_sequence = sequences[-sequence_length:].reshape(1, -1, 1)

# Create an array to store future predictions
future_predictions = []

# Predict 7 days in the future
for _ in range(7):
    prediction = model.predict(input_sequence)
    future_predictions.append(prediction)
    input_sequence = np.roll(input_sequence, -1, axis=1)
    input_sequence[0, -1, 0] = prediction

# Inverse transform the predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Flatten the future_predictions array
future_predictions_flattened = future_predictions.flatten()

# Calculate MAE and RMSE
mae = mean_absolute_error(actual_rain_data, future_predictions_flattened)
rmse = mean_squared_error(actual_rain_data, future_predictions_flattened, squared=False)

print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)

# Create a graph
plt.figure(figsize=(10, 6))
plt.plot(actual_rain_data[:7], label='Actual Rain', marker='o')
plt.plot(future_predictions_flattened[:7], label='Predicted Rain', marker='x')

# Draw lines between each actual and predicted point
for i in range(7):
    actual_rain = actual_rain_data[i]  # Get the actual rain for the corresponding day
    plt.plot([i, i], [actual_rain, future_predictions_flattened[i]], color='red', linestyle='--')

plt.xlabel('Day')
plt.ylabel('Rain')
plt.title('Actual vs. Predicted Rain')
plt.legend()
plt.grid(True)
plt.show()