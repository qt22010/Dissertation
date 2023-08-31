# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 02:41:44 2023

@author: white
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Load your merged dataset
merged_dataset = pd.read_csv('merged_dataset.csv')  # Replace with your dataset file

# Convert date column to datetime format
merged_dataset['Date1'] = pd.to_datetime(merged_dataset['Date1'])

# Define the features and target variable
features = ['WindSpeed', 'WindDirection', 'O3', 'PM2.5', 'PM10']
target = 'PollenCount'

# Create lag features for sequences
sequence_length = 7  # Number of past days to consider for prediction
X_sequences = []
y_sequences = []

for i in range(len(merged_dataset) - sequence_length - 7):
    X_sequences.append(merged_dataset[features].iloc[i:i+sequence_length])
    y_sequences.append(merged_dataset[target].iloc[i+sequence_length+7])  # Predict 7 days ahead

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the LSTM model
def build_lstm_model(units=64, num_layers=2, optimizer='adam', activation='relu'):
    model = keras.Sequential()
    model.add(layers.Input(shape=(sequence_length, len(features))))
    for _ in range(num_layers):
        model.add(layers.LSTM(units, activation=activation, return_sequences=True))
    model.add(layers.LSTM(units, activation=activation))
    model.add(layers.Dense(1))  # Output layer
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])
    return model

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'units': [32, 64, 128],
    'num_layers': [1, 2, 3],
    'batch_size': [32, 64],
    'epochs': [50, 100],
    'optimizer': ['adam', 'rmsprop'],
    'activation': ['relu', 'tanh', 'sigmoid']
}

lstm_model = KerasRegressor(build_fn=build_lstm_model, verbose=0)
grid_search = GridSearchCV(estimator=lstm_model, param_grid=param_grid, cv=3, verbose=2)
grid_result = grid_search.fit(X_train_scaled, y_train)

best_lstm_model = grid_result.best_estimator_.model

# Make predictions
predictions = best_lstm_model.predict(X_test_scaled)

r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)
f1 = f1_score(y_test, np.where(predictions > 0.5, 1, 0), average='macro')  # Assuming binary classification

print("R-squared (R2):", r2)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("F1 Score:", f1)