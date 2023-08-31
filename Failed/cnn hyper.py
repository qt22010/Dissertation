# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

# Load and preprocess data (assuming you've loaded the data as before)
# ...

# Define the look-ahead period (1 week)
look_ahead = 7

# Create lag features for sequences
# ...

# Split data into train and test sets
# ...

# Standardize features
# ...

def build_model(hp):
    model = Sequential()
    model.add(Conv1D(filters=hp.Int('filters', min_value=32, max_value=128, step=32), kernel_size=3, activation='relu', input_shape=(look_ahead, X.shape[1])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))  # For temperature regression

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='mean_squared_error')
    return model

# Define the Keras Tuner RandomSearch for temperature prediction
random_temp_tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,  # Number of combinations to try
    directory='random_temp_dir',  # Directory to save results
    project_name='temperature_random_tuning'  # Name for this tuning task
)

# Search for the best hyperparameters using Random Search for temperature prediction
random_temp_tuner.search(X_train_scaled, y_temp_train, epochs=10, validation_split=0.1, verbose=1)

# Get the best model's hyperparameters for temperature prediction
random_temp_best_hp = random_temp_tuner.get_best_hyperparameters()[0]

# Build and compile the final model for temperature prediction with the best hyperparameters from Random Search
random_temp_final_model = random_temp_tuner.hypermodel.build(random_temp_best_hp)
random_temp_final_model.compile(optimizer=keras.optimizers.Adam(learning_rate=random_temp_best_hp.get('learning_rate')),
                               loss='mean_squared_error')

# Train the final model for temperature prediction with the best hyperparameters from Random Search
random_temp_history = random_temp_final_model.fit(X_train_scaled, y_temp_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Evaluate the final model for temperature prediction
random_temp_predictions = random_temp_final_model.predict(X_test_scaled).flatten()
random_temp_mse = mean_squared_error(y_temp_test, random_temp_predictions)
print(f'Mean Squared Error on Test Data (Random Search - Temperature): {random_temp_mse:.2f}')

# Plot training and validation loss for temperature prediction
plt.figure(figsize=(10, 6))
plt.plot(random_temp_history.history['loss'], label='Training Loss')
plt.plot(random_temp_history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss (Temperature)')
plt.legend()
plt.show()

# Define the Keras Tuner RandomSearch for precipitation prediction
random_precip_tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,  # Number of combinations to try
    directory='random_precip_dir',  # Directory to save results
    project_name='precipitation_random_tuning'  # Name for this tuning task
)

# Search for the best hyperparameters using Random Search for precipitation prediction
random_precip_tuner.search(X_train_scaled, y_precip_train, epochs=10, validation_split=0.1, verbose=1)

# Get the best model's hyperparameters for precipitation prediction
random_precip_best_hp = random_precip_tuner.get_best_hyperparameters()[0]

# Build and compile the final model for precipitation prediction with the best hyperparameters from Random Search
random_precip_final_model = random_precip_tuner.hypermodel.build(random_precip_best_hp)
random_precip_final_model.compile(optimizer=keras.optimizers.Adam(learning_rate=random_precip_best_hp.get('learning_rate')),
                                 loss='binary_crossentropy')

# Train the final model for precipitation prediction with the best hyperparameters from Random Search
random_precip_history = random_precip_final_model.fit(X_train_scaled, y_precip_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Evaluate the final model for precipitation prediction
random_precip_predictions = random_precip_final_model.predict(X_test_scaled).flatten()
random_precip_mse = mean_squared_error(y_precip_test, random_precip_predictions)
print(f'Mean Squared Error on Test Data (Random Search - Precipitation): {random_precip_mse:.2f}')

# Plot training and validation loss for precipitation prediction
plt.figure(figsize=(10, 6))
plt.plot(random_precip_history.history['loss'], label='Training Loss')
plt.plot(random_precip_history.history['val_loss'], label='Validation Loss')
plt.xlabel
