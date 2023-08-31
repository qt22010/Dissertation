# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 13:33:47 2023

@author: white
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from math import sqrt

# Load your dataset and preprocess as needed
data = pd.read_csv(r'C:\Users\white\Desktop\Dissertation\data\temp_weather.csv')

# Define the sequence length (n-1 days)
sequence_length = 7

features = ['dew_point', 'temp_min', 'temp_max', 'humidity', 'clouds_all', 'snow_1h', 'day_of_week', 'month', 'year', 'season']
target_temp = 'temp'  # Target variable for temperature prediction

# Splitting the data into features and targets
X = data[features]
y = data[target_temp]

# Create sequences and corresponding target values
X_sequences = []
y_sequences = []

for i in range(len(data) - sequence_length - 7):
    X_sequences.append(X.iloc[i:i+sequence_length].values)
    y_sequences.append(y.iloc[i+sequence_length:i+sequence_length+7].values)

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(sequence_length, len(features))))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(7))  # Output layer for 7-day forecast

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Make predictions
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)

# Calculate the R-squared score
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared Score: {r2}")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from numba import jit
# Load your dataset and preprocess as needed
data = pd.read_csv(r'C:\Users\white\Desktop\Dissertation\data\temp_weather.csv')

# Define the sequence length (n-1 days)
sequence_length = 7

features = ['dew_point', 'temp_min', 'temp_max', 'humidity', 'clouds_all', 'snow_1h', 'day_of_week', 'month', 'year', 'season']
target_temp = 'temp'  # Target variable for temperature prediction

# Splitting the data into features and targets
X = data[features]
y = data[target_temp]

# Create sequences and corresponding target values
X_sequences = []
y_sequences = []

for i in range(len(data) - sequence_length - 7):
    X_sequences.append(X.iloc[i:i+sequence_length].values)
    y_sequences.append(y.iloc[i+sequence_length:i+sequence_length+7].values)

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)

# Function to build the CNN model
def build_model(filters, units, learning_rate, dropout_rate, num_conv_layers):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=3, activation='relu', padding='same', input_shape=X_train.shape[1:]))
    model.add(MaxPooling1D(pool_size=2))
    for _ in range(num_conv_layers - 1):
        model.add(Conv1D(filters=filters, kernel_size=3, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(7))

    model.compile(
        loss=MeanSquaredError(),
        optimizer=Adam(learning_rate=learning_rate),
        metrics=[RootMeanSquaredError()]
    )
    
    return model

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'filters': [16, 32, 64],
    'units': [32, 64, 128],
    'learning_rate': [1e-2, 1e-3, 1e-4],
    'dropout_rate': [0.2, 0.4, 0.6],
    'num_conv_layers': [1, 2, 3]
}

best_mse = float('inf')
best_params = {}

# Hyperparameter tuning loop
for filters in param_grid['filters']:
    for units in param_grid['units']:
        for learning_rate in param_grid['learning_rate']:
            for dropout_rate in param_grid['dropout_rate']:
                for num_conv_layers in param_grid['num_conv_layers']:
                    model = build_model(filters, units, learning_rate, dropout_rate, num_conv_layers)
                    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    
                    if mse < best_mse:
                        best_mse = mse
                        best_params = {
                            'filters': filters,
                            'units': units,
                            'learning_rate': learning_rate,
                            'dropout_rate': dropout_rate,
                            'num_conv_layers': num_conv_layers
                        }

print("Best Hyperparameters:", best_params)
print("Best Validation MSE:", best_mse)

# Calculate R-squared score for the best model
# Build the best model using the best parameters
best_model = build_model(**best_params)

# Train the best model on the entire training set
best_history = best_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])

y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R-squared Score: {r2}")

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Use the best model to make predictions
y_pred = best_model.predict(X_test)

# Plot true vs. predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs. Predicted Values')
plt.show()

# Plot histogram of residuals
residuals = y_test - y_pred
plt.hist(residuals, bins=30)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.show()

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared Score: {r2}")

# Check the number of available GPUs
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available:", len(physical_devices))

import tensorflow as tf
tf.test.gpu_device_name()