# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 09:14:05 2023

@author: white
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import tensorflow as tf

# Load and preprocess data
data = pd.read_csv(r'C:\Users\white\Desktop\Dissertation\data\weather_data_cleaned1.csv')
features = ['dew_point', 'feels_like', 'temp_min', 'temp_max', 'pressure', 'humidity', 'wind_speed', 'wind_deg', 'clouds_all', 'rain_1h', 'snow_1h']
target_temp = 'temp'
X = data[features]
y = data[target_temp]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape the data to be suitable for the CNN model
look_back = 11  # Adjust this based on your preferred sequence length
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build a CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(look_back, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Enable TensorFlow GPU acceleration (if available)
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Train the model
model.fit(X_train_cnn, y_train, epochs=50, batch_size=32)

# Evaluate the model
mse = model.evaluate(X_test_cnn, y_test)
print(f'Mean Squared Error: {mse}')

# Make future predictions
input_sequence = X_test_cnn[-1]  # Use the last sequence from the testing set

# Create an array to store future predictions
future_predictions = []

# Predict 7 days in the future
for _ in range(7):
    prediction = model.predict(np.array([input_sequence]))  # Predict the next day
    future_predictions.append(prediction)
    input_sequence = np.roll(input_sequence, -1)  # Shift the input sequence
    input_sequence[-1] = prediction  # Update the last element with the new prediction

# Convert the list of predictions to a NumPy array
future_predictions = np.array(future_predictions)

print("Predicted temperatures for the next 7 days:")
print(future_predictions)

actual_temperature_data = [15.77, 16.43, 16.55, 19.19, 19.36, 19.17, 17.38]


import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Calculate MAE and RMSE
mae = mean_absolute_error(actual_temperature_data, future_predictions.flatten())
rmse = mean_squared_error(actual_temperature_data, future_predictions.flatten(), squared=False)

print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)


# Flatten the future_predictions array
future_predictions_flattened = future_predictions.flatten()

# Create a graph
plt.figure(figsize=(10, 6))
plt.plot(actual_temperature_data, label='Actual Temperatures', marker='o')
plt.plot(future_predictions_flattened, label='Predicted Temperatures', marker='x')

# Draw lines between each actual and predicted point
for i in range(len(actual_temperature_data)):
    plt.plot([i, i], [actual_temperature_data[i], future_predictions_flattened[i]], color='red', linestyle='--')

plt.xlabel('Day')
plt.ylabel('Temperature')
plt.title('Actual vs. Predicted Temperatures')
plt.legend()
plt.grid(True)
plt.show()

features = ['dew_point', 'feels_like', 'temp_min', 'temp_max', 'pressure', 'humidity', 'wind_speed', 'wind_deg', 'clouds_all', 'temp', 'snow_1h']
target_rain = 'rain_1h'
X = data[features]
y = data[target_rain]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape the data to be suitable for the CNN model
look_back = 11  # Adjust this based on your preferred sequence length
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build a CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(look_back, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Enable TensorFlow GPU acceleration (if available)
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Train the model
model.fit(X_train_cnn, y_train, epochs=50, batch_size=32)

# Evaluate the model
mse = model.evaluate(X_test_cnn, y_test)
print(f'Mean Squared Error: {mse}')

# Make future predictions
input_sequence = X_test_cnn[-1]  # Use the last sequence from the testing set

# Create an array to store future predictions
future_predictions = []

# Predict 7 days in the future
for _ in range(7):
    prediction = model.predict(np.array([input_sequence]))  # Predict the next day
    future_predictions.append(prediction)
    input_sequence = np.roll(input_sequence, -1)  # Shift the input sequence
    input_sequence[-1] = prediction  # Update the last element with the new prediction

# Convert the list of predictions to a NumPy array
future_predictions = np.array(future_predictions)

print("Predicted rain for the next 7 days:")
print(future_predictions)

actual_rain_data = [2.21, 0, 11.39, 5.01, 0, 0.26, 10.16]


import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Calculate MAE and RMSE
mae = mean_absolute_error(actual_rain_data, future_predictions.flatten())
rmse = mean_squared_error(actual_rain_data, future_predictions.flatten(), squared=False)

print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)


# Flatten the future_predictions array
future_predictions_flattened = future_predictions.flatten()

# Create a graph
plt.figure(figsize=(10, 6))
plt.plot(actual_rain_data, label='Actual Rain', marker='o')
plt.plot(future_predictions_flattened, label='Predicted Rain', marker='x')

# Draw lines between each actual and predicted point
for i in range(len(actual_rain_data)):
    plt.plot([i, i], [actual_rain_data[i], future_predictions_flattened[i]], color='red', linestyle='--')

plt.xlabel('Day')
plt.ylabel('Rain')
plt.title('Actual vs. Predicted Rain')
plt.legend()
plt.grid(True)
plt.show()