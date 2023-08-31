# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 09:46:21 2023

@author: white
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

# Load and preprocess data
data = pd.read_csv('weather_data.csv')
X = data[['mean_temp', 'max_temp', 'min_temp', 'pressure', 'clouds', 'dew_point', 'wind_speed', 'wind_direction', 'humidity', 'snow']]
y_temp = data['mean_temp']
y_precip = data['rain']

# Define the look-ahead period (1 week)
look_ahead = 7

# Create sequences for RNN
X_sequences = []
y_sequences_temp = []
y_sequences_precip = []

for i in range(len(X) - look_ahead):
    X_sequences.append(X.iloc[i:i+look_ahead].values)
    y_sequences_temp.append(y_temp.iloc[i+look_ahead])
    y_sequences_precip.append(y_precip.iloc[i+look_ahead])

X_sequences = np.array(X_sequences)
y_sequences_temp = np.array(y_sequences_temp)
y_sequences_precip = np.array(y_sequences_precip)

# Split data into train and test sets
X_train, X_test, y_temp_train, y_temp_test, y_precip_train, y_precip_test = train_test_split(
    X_sequences, y_sequences_temp, y_sequences_precip, test_size=0.1, random_state=42, shuffle=False
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[2]))
X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[2]))

X_train_scaled = X_train_scaled.reshape(X_train.shape)
X_test_scaled = X_test_scaled.reshape(X_test.shape)

# Define a function to create the LSTM model with Attention
def create_lstm_attention_model(units=32, dropout_rate=0.2, num_layers=1, attention_units=16):
    model = Sequential()
    for _ in range(num_layers):
        model.add(LSTM(units, activation='relu', return_sequences=True))
        model.add(Dropout(dropout_rate))
    model.add(LSTM(units // 2, activation='relu', return_sequences=True))
    model.add(Attention(use_scale=True, units=attention_units))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))  # Two output nodes for temperature regression and precipitation classification
    model.compile(optimizer='adam', loss='mean_squared_error', loss_weights=[0.5, 1])  # Adjust loss weights as needed
    return model

# Create KerasRegressor for LSTM model with Attention
lstm_attention_regressor = KerasRegressor(build_fn=create_lstm_attention_model, verbose=0)

# Define hyperparameters for Grid Search
param_grid = {
    'units': [16, 32, 64],
    'dropout_rate': [0.2, 0.3, 0.4],
    'num_layers': [1, 2, 3],
    'attention_units': [8, 16, 32]
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=lstm_attention_regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
grid_result = grid_search.fit(X_train_scaled, [y_temp_train, y_precip_train])

# Get the best hyperparameters from the grid search
best_units_grid = grid_result.best_params_['units']
best_dropout_rate_grid = grid_result.best_params_['dropout_rate']
best_num_layers_grid = grid_result.best_params_['num_layers']
best_attention_units_grid = grid_result.best_params_['attention_units']

print("Best Hyperparameters from Grid Search:")
print(f'Units: {best_units_grid}')
print(f'Dropout Rate: {best_dropout_rate_grid}')
print(f'Number of Layers: {best_num_layers_grid}')
print(f'Attention Units: {best_attention_units_grid}')

# Create the LSTM model with best hyperparameters from Grid Search
best_lstm_attention_model = create_lstm_attention_model(units=best_units_grid, dropout_rate=best_dropout_rate_grid, num_layers=best_num_layers_grid, attention_units=best_attention_units_grid)

# Train the model for temperature and precipitation predictions
history_best_lstm_attention = best_lstm_attention_model.fit(X_train_scaled, [y_temp_train, y_precip_train], epochs=50, batch_size=32, verbose=0)

# Evaluate the best LSTM Attention model
predictions_temp, predictions_precip_prob = best_lstm_attention_model.predict(X_test_scaled)
predictions_precip = np.where(predictions_precip_prob > 0.5, 1, 0)

# Evaluate metrics for Temperature
mse_temp = mean_squared_error(y_temp_test, predictions_temp[:, 0])
rmse_temp = np.sqrt(mse_temp)
mae_temp = mean_absolute_error(y_temp_test, predictions_temp[:, 0])
r2_temp = r2_score(y_temp_test, predictions_temp[:, 0])
f1_score_temp = f1_score(y_temp_test, np.round(predictions_temp[:, 0]))

# Evaluate metrics for Precipitation
accuracy_precip = np.mean(predictions_precip[:, 1] == y_precip_test)
f1_score_precip = f1_score(y_precip_test, predictions_precip[:, 1])
mse_precip = mean_squared_error(y_precip_test, predictions_precip_prob[:, 1])
rmse_precip = np.sqrt(mse_precip)
mae_precip = mean_absolute_error(y_precip_test, predictions_precip_prob[:, 1])
r2_precip = r2_score(y_precip_test, predictions_precip_prob[:, 1])

# Print evaluation metrics for Temperature
print("Metrics for Temperature Prediction (LSTM with Attention):")
print(f'R-squared (R2): {r2_temp:.2f}')
print(f'Mean Squared Error (MSE): {mse_temp:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse_temp:.2f}')
print(f'Mean Absolute Error (MAE): {mae_temp:.2f}')
print(f'F1 Score: {f1_score_temp:.2f}')

# Print evaluation metrics for Precipitation
print("\nMetrics for Precipitation Prediction (LSTM with Attention):")
print(f'Accuracy: {accuracy_precip:.2f}')
print(f'F1 Score: {f1_score_precip:.2f}')
print(f'Mean Squared Error (MSE): {mse_precip:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse_precip:.2f}')
print(f'Mean Absolute Error (MAE): {mae_precip:.2f}')
print(f'R-squared (R2): {r2_precip:.2f}')

# Create plots for Temperature predictions
plt.figure(figsize=(10, 6))
plt.plot(y_temp_test, label='Actual Temperature', marker='o')
plt.plot(predictions_temp[:, 0], label='Predicted Temperature', marker='o')
plt.xlabel('Days')
plt.ylabel('Temperature')
plt.title('Actual vs. Predicted Temperature')
plt.legend()
plt.show()

# Create plots for Precipitation predictions
plt.figure(figsize=(10, 6))
plt.plot(y_precip_test, label='Actual Precipitation', marker='o')
plt.plot(predictions_precip_prob[:, 1], label='Predicted Precipitation', marker='o')
plt.xlabel('Days')
plt.ylabel('Precipitation Probability')
plt.title('Actual vs. Predicted Precipitation Probability')
plt.legend()
plt.show()
