import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate

# Define the actual temperature and rain data (from a separate list)
actual_temperature_data = [15.77, 16.43, 16.55, 19.19, 19.36, 19.17, 17.38]
actual_rain_data = [2.21, 0, 11.39, 5.01, 0, 0.26, 10.16]

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

# Set the length of input sequences (adjust as needed)
input_sequence_length = 30

# Create input sequences for temperature and rain
sequences_temp = []
sequences_rain = []
for i in range(len(y_scaled_temp) - input_sequence_length):
    sequences_temp.append(y_scaled_temp[i:i+input_sequence_length])
    sequences_rain.append(y_scaled_rain[i:i+input_sequence_length])
sequences_temp = np.array(sequences_temp)
sequences_rain = np.array(sequences_rain)

# Prepare target values (starting from where input sequences end)
target_values_temp = y_scaled_temp[input_sequence_length:]
target_values_rain = y_scaled_rain[input_sequence_length:]

# Build the combined LSTM model for temperature and rain using Functional API
input_temp = Input(shape=(input_sequence_length, 1))
input_rain = Input(shape=(input_sequence_length, 1))
lstm_temp = LSTM(units=64, activation='relu')(input_temp)
lstm_rain = LSTM(units=64, activation='relu')(input_rain)
concatenated = Concatenate()([lstm_temp, lstm_rain])
output = Dense(2)(concatenated)
model = Model(inputs=[input_temp, input_rain], outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit([sequences_temp, sequences_rain], target_values_temp, epochs=50, batch_size=32)

# Create input sequences for the next 7 days (not present in the dataset)
input_sequence_temp = y_scaled_temp[-input_sequence_length:].reshape(1, input_sequence_length, 1)
input_sequence_rain = y_scaled_rain[-input_sequence_length:].reshape(1, input_sequence_length, 1)

# Create arrays to store future predictions
future_predictions_temp = []
future_predictions_rain = []

# Predict 7 days in the future for both temperature and rain
for _ in range(7):
    predictions = model.predict([input_sequence_temp, input_sequence_rain])
    prediction_temp = predictions[0][0]  # Temperature prediction
    prediction_rain = predictions[0][1]  # Rain prediction
    future_predictions_temp.append(prediction_temp)
    future_predictions_rain.append(prediction_rain)
    
    input_sequence_temp = np.roll(input_sequence_temp, -1, axis=1)
    input_sequence_temp[0, -1, 0] = prediction_temp
    
    input_sequence_rain = np.roll(input_sequence_rain, -1, axis=1)
    input_sequence_rain[0, -1, 0] = prediction_rain

# Inverse transform the predictions
future_predictions_temp = scaler_temp.inverse_transform(np.array(future_predictions_temp).reshape(-1, 1))
future_predictions_rain = scaler_rain.inverse_transform(np.array(future_predictions_rain).reshape(-1, 1))

# Calculate MAE and RMSE for temperature and rain
mae_temp = mean_absolute_error(actual_temperature_data, future_predictions_temp)
rmse_temp = mean_squared_error(actual_temperature_data, future_predictions_temp, squared=False)
mae_rain = mean_absolute_error(actual_rain_data, future_predictions_rain)
rmse_rain = mean_squared_error(actual_rain_data, future_predictions_rain, squared=False)

# Print the errors
print("Temperature - Mean Absolute Error:", mae_temp)
print("Temperature - Root Mean Squared Error:", rmse_temp)
print("Rain - Mean Absolute Error:", mae_rain)
print("Rain - Root Mean Squared Error:", rmse_rain)

# Create a graph for temperature and rain predictions
plt.figure(figsize=(10, 6))
plt.plot(actual_temperature_data[:7], label='Actual Temperatures', marker='o')
plt.plot(future_predictions_temp[:7], label='Predicted Temperatures', marker='x')
plt.plot(actual_rain_data[:7], label='Actual Rain', marker='s')
plt.plot(future_predictions_rain[:7], label='Predicted Rain', marker='d')

# Draw lines between each actual and predicted point for temperature
for i in range(7):
    actual_temp_day = actual_temperature_data[i]  # Get the actual temperature for the corresponding day
    plt.plot([i, i], [actual_temp_day, future_predictions_temp[i]], color='red', linestyle='--')

# Draw lines between each actual and predicted point for rain
for i in range(7):
    actual_rain_day = actual_rain_data[i]  # Get the actual rain for the corresponding day
    plt.plot([i, i], [actual_rain_day, future_predictions_rain[i]], color='blue', linestyle='--')

# Add labels and title
plt.xlabel('Day')
plt.title('Actual vs. Predicted Temperatures and Rain')
plt.legend()
plt.grid(True)
plt.show()