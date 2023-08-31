# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 11:29:29 2023

@author: white
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.models import Model
import tensorflow as tf

# Load and preprocess data
data = pd.read_csv(r'C:\Users\white\Desktop\Dissertation\data\weather_data_cleaned1.csv')
features = ['dew_point', 'feels_like', 'temp_min', 'temp_max', 'pressure', 'humidity', 'wind_speed', 'wind_deg', 'clouds_all', 'rain_1h', 'snow_1h']
target_temp = 'temp'
target_rain = 'rain_1h'
X = data[features]
y_temp = data[target_temp]
y_rain = data[target_rain]

actual_temperature_data = [15.77, 16.43, 16.55, 19.19, 19.36, 19.17, 17.38]
actual_rain_data = [2.21, 0, 11.39, 5.01, 0, 0.26, 10.16]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_temp_train, y_temp_test, y_rain_train, y_rain_test = train_test_split(X_scaled, y_temp, y_rain, test_size=0.2, random_state=42)

# Reshape the data to be suitable for the CNN model
look_back = 11  # Adjust this based on your preferred sequence length
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build a multi-output CNN model
input_layer = Input(shape=(look_back, 1))
conv_layer = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
pooling_layer = MaxPooling1D(pool_size=2)(conv_layer)
flatten_layer = Flatten()(pooling_layer)
dense_layer = Dense(50, activation='relu')(flatten_layer)
output_temp = Dense(1)(dense_layer)  # Output for temperature prediction
output_rain = Dense(1)(dense_layer)  # Output for rain prediction

model = Model(inputs=input_layer, outputs=[output_temp, output_rain])
model.compile(optimizer='adam', loss='mean_squared_error')

# Enable TensorFlow GPU acceleration (if available)
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Train the model
model.fit(X_train_cnn, [y_temp_train, y_rain_train], epochs=50, batch_size=32)

# Evaluate the model
loss = model.evaluate(X_test_cnn, [y_temp_test, y_rain_test])
print(f'Total Loss: {loss}')

# Make future predictions
input_sequence = X_test_cnn[-1]  # Use the last sequence from the testing set

# Create arrays to store future predictions
future_predictions_temp = []
future_predictions_rain = []

# Predict 7 days in the future
for _ in range(7):
    predictions = model.predict(np.array([input_sequence]))  # Predict for both temperature and rain
    future_predictions_temp.append(predictions[0][0])
    future_predictions_rain.append(predictions[1][0])
    input_sequence = np.roll(input_sequence, -1)  # Shift the input sequence
    input_sequence[-1] = predictions[0][0]  # Update the last element with the new prediction

print("Predicted temperatures for the next 7 days:")
print(future_predictions_temp)
print("Predicted rain for the next 7 days:")
print(future_predictions_rain)

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Calculate MAE and RMSE for temperature predictions
mae_temp = mean_absolute_error(actual_temperature_data, future_predictions_temp)
rmse_temp = mean_squared_error(actual_temperature_data, future_predictions_temp, squared=False)

print("Temperature Mean Absolute Error:", mae_temp)
print("Temperature Root Mean Squared Error:", rmse_temp)

# Calculate MAE and RMSE for rain predictions
mae_rain = mean_absolute_error(actual_rain_data, future_predictions_rain)
rmse_rain = mean_squared_error(actual_rain_data, future_predictions_rain, squared=False)

print("Rain Mean Absolute Error:", mae_rain)
print("Rain Root Mean Squared Error:", rmse_rain)

# Flatten the future_predictions arrays
future_predictions_temp_flattened = np.array(future_predictions_temp).flatten()
future_predictions_rain_flattened = np.array(future_predictions_rain).flatten()

# Create temperature and rain graphs
plt.figure(figsize=(10, 6))

# Temperature graph
plt.subplot(2, 1, 1)
plt.plot(actual_temperature_data, label='Actual Temperatures', marker='o')
plt.plot(future_predictions_temp_flattened, label='Predicted Temperatures', marker='x')

# Draw lines between each actual and predicted point for temperature
for i in range(len(actual_temperature_data)):
    plt.plot([i, i], [actual_temperature_data[i], future_predictions_temp_flattened[i]], color='red', linestyle='--')

plt.ylabel('Temperature')
plt.title('Actual vs. Predicted Temperatures')
plt.legend()
plt.grid(True)

# Rain graph
plt.subplot(2, 1, 2)
plt.plot(actual_rain_data, label='Actual Rain', marker='o')
plt.plot(future_predictions_rain_flattened, label='Predicted Rain', marker='x')

# Draw lines between each actual and predicted point for rain
for i in range(len(actual_rain_data)):
    plt.plot([i, i], [actual_rain_data[i], future_predictions_rain_flattened[i]], color='red', linestyle='--')

plt.xlabel('Day')
plt.ylabel('Rain')
plt.title('Actual vs. Predicted Rain')
plt.legend()
plt.grid(True)

# Adjust layout and show the plots
plt.tight_layout()
plt.show()