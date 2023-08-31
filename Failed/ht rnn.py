# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 05:44:32 2023

@author: white
"""

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout

# Load and preprocess data
# ... (Previous code for data loading)

# Define a function to create the advanced RNN model
def create_advanced_rnn_model(units=32, dropout_rate=0.2, num_layers=1):
    model = Sequential()
    for _ in range(num_layers):
        model.add(SimpleRNN(units, activation='relu', return_sequences=True))
        model.add(Dropout(dropout_rate))
    model.add(SimpleRNN(units // 2, activation='relu', return_sequences=False))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))  # Two output nodes for temperature regression and precipitation classification
    model.compile(optimizer='adam', loss='mean_squared_error', loss_weights=[0.5, 1])  # Adjust loss weights as needed
    return model

# Create KerasRegressor for advanced RNN model
advanced_rnn_regressor = KerasRegressor(build_fn=create_advanced_rnn_model, verbose=0)

# Define hyperparameters to tune BIGGER THE RANGE PLEASE
param_grid = {
    'units': [16, 32, 64],
    'dropout_rate': [0.2, 0.3, 0.4],
    'num_layers': [1, 2, 3]
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=advanced_rnn_regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
grid_result = grid_search.fit(X_train_scaled, [y_temp_train, y_precip_train])

# Get the best hyperparameters from the grid search
best_units = grid_result.best_params_['units']
best_dropout_rate = grid_result.best_params_['dropout_rate']
best_num_layers = grid_result.best_params_['num_layers']

print("Best Hyperparameters:")
print(f'Units: {best_units}')
print(f'Dropout Rate: {best_dropout_rate}')
print(f'Number of Layers: {best_num_layers}')

# Create the advanced RNN model with best hyperparameters
best_advanced_rnn_model = create_advanced_rnn_model(units=best_units, dropout_rate=best_dropout_rate, num_layers=best_num_layers)

# Train the model for temperature and precipitation predictions
history_best_advanced_rnn = best_advanced_rnn_model.fit(X_train_scaled, [y_temp_train, y_precip_train], epochs=50, batch_size=32, verbose=0)

# Plot Training Loss for the best Advanced RNN model
plt.figure(figsize=(10, 6))
plt.plot(history_best_advanced_rnn.history['loss'], label='Total Loss')
plt.plot(history_best_advanced_rnn.history['dense_1_loss'], label='Temperature Loss')
plt.plot(history_best_advanced_rnn.history['dense_2_loss'], label='Precipitation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss for Best Advanced RNN Model')
plt.legend()
plt.show()

"""
# Evaluate the best Advanced RNN model
# ... (Rest of the evaluation code for temperature and precipitation predictions)
Explanation of Changes Made:

Number of Layers Optimization: We added the hyperparameter num_layers to the param_grid. In the create_advanced_rnn_model function, we modified the loop to add multiple layers based on the num_layers value. This allows us to explore the impact of different numbers of layers in the RNN architecture.

Hyperparameter Grid: The param_grid now includes options for num_layers. We also kept the options for units and dropout_rate to continue tuning those hyperparameters.

Printing Best Hyperparameters: After the grid search, we print the best hyperparameters found, including the number of layers.

Model Creation: We create the best advanced RNN model using the best hyperparameters found in the grid search.

By optimizing the number of layers along with other hyperparameters, you're exploring a wider range of architectures to find the optimal model configuration for your data. This approach allows you to systematically experiment and select the architecture that performs best on your specific problem.
"""