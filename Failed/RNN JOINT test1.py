# -*- coding: utf-8 -*-

actual_temperature_data = [15.77, 16.43, 16.55, 19.19, 19.36, 19.17, 17.38]
actual_rain_data = [2.21, 0, 11.39, 5.01, 0, 0.26, 10.16]

actual_temp=actual_temperature_data
actual_rain=actual_rain_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Load and preprocess data
data = pd.read_csv(r'C:\Users\white\Desktop\Dissertation\data\weather_data_cleaned1.csv')  # Replace with the actual path
target_temp = 'temp'
target_rain = 'rain_1h'
y_temp = data[target_temp]
y_rain = data[target_rain]

# Normalize the target data
scaler_temp = StandardScaler()
scaler_rain = StandardScaler()
y_scaled_temp = scaler_temp.fit_transform(np.array(y_temp).reshape(-1, 1))
y_scaled_rain = scaler_rain.fit_transform(np.array(y_rain).reshape(-1, 1))

# Create sequences for RNN using all of the available data
sequence_length = len(y_scaled_temp)
sequences_temp = y_scaled_temp
sequences_rain = y_scaled_rain

# Split the data into training and testing sets
X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
    sequences_temp, y_scaled_temp, test_size=0.2, shuffle=False
)
X_train_rain, X_test_rain, y_train_rain, y_test_rain = train_test_split(
    sequences_rain, y_scaled_rain, test_size=0.2, shuffle=False
)

# Build the combined RNN model for temperature and rain
model = Sequential()
model.add(SimpleRNN(units=64, activation='relu', input_shape=(sequence_length, 1)))
model.add(Dense(2))  # Two outputs: one for temperature, one for rain
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train_temp, np.column_stack((y_train_temp, y_train_rain)), epochs=50, batch_size=32)

# Make predictions using the testing data for temperature
test_predictions = model.predict(X_test_temp)
test_predictions_temp = scaler_temp.inverse_transform(test_predictions)[:, 0]  # Only temperature predictions

# Calculate MAE and RMSE for testing data
mae_test_temp = mean_absolute_error(y_test_temp, test_predictions_temp)
rmse_test_temp = mean_squared_error(y_test_temp, test_predictions_temp, squared=False)

print("Testing - Temperature - Mean Absolute Error:", mae_test_temp)
print("Testing - Temperature - Root Mean Squared Error:", rmse_test_temp)

# Make predictions using the testing data for rain
test_predictions_rain = scaler_rain.inverse_transform(test_predictions)[:, 1]  # Only rain predictions

# Calculate MAE and RMSE for rain predictions on testing data
mae_test_rain = mean_absolute_error(y_test_rain, test_predictions_rain)
rmse_test_rain = mean_squared_error(y_test_rain, test_predictions_rain, squared=False)

print("Testing - Rain - Mean Absolute Error:", mae_test_rain)
print("Testing - Rain - Root Mean Squared Error:", rmse_test_rain)

plt.figure(figsize=(10, 6))
plt.plot(actual_temp[:7], label='Actual Temperatures', marker='o')
plt.plot(test_predictions_temp[:7], label='Predicted Temperatures', marker='x')
plt.plot(actual_rain[:7], label='Actual Rain', marker='s')
plt.plot(test_predictions_rain[:7], label='Predicted Rain', marker='d')

# Draw lines between each actual and predicted point for temperature
for i in range(7):
    actual_temp_day = actual_temp[i]  # Get the actual temperature for the corresponding day
    plt.plot([i, i], [actual_temp_day, test_predictions_temp[i]], color='red', linestyle='--')

# Draw lines between each actual and predicted point for rain
for i in range(7):
    actual_rain_day = actual_rain[i]  # Get the actual rain for the corresponding day
    plt.plot([i, i], [actual_rain_day, test_predictions_rain[i]], color='blue', linestyle='--')

plt.xlabel('Day')
plt.title('Testing - Actual vs. Predicted Temperatures and Rain')
plt.legend()
plt.grid(True)
plt.show()
print("Actual Temperatures:", actual_temp)