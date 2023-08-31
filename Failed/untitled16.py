# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler

# Load your dataset and preprocess as needed
data = pd.read_csv('temp_weather.csv')

# Define the sequence length (n-1 days)
sequence_length = 7

features = ['dew_point', 'temp_min', 'temp_max', 'humidity', 'clouds_all', 'snow_1h', 'day_of_week', 'month', 'year', 'season']
target_temp = 'temp'  # Target variable for temperature prediction

# Splitting the data into features and targets
X = data[features]
y = data[target_temp]

# Standardize the features using Z-Score Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create sequences and corresponding target values
X_sequences = []
y_sequences = []

for i in range(len(data) - sequence_length - 7):
    X_sequences.append(X_scaled[i:i+sequence_length])
    y_sequences.append(y.iloc[i+sequence_length:i+sequence_length+7].values)

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

# Define the number of splits for time-series cross-validation
n_splits = 3

# Create a TimeSeriesSplit object
tscv = TimeSeriesSplit(n_splits=n_splits)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'units': [32, 64, 128],
    'num_nodes_per_layer': [[32], [64], [128], [64, 64], [128, 128]],
    'learning_rate': [1e-2, 1e-3, 1e-4],
    'dropout_rate': [0, 0.2, 0.4, 0.6],  # Dropout rate for input layer (0 for no dropout)
    'dropout_rate_lstm': [0, 0.2, 0.4, 0.6],   # Dropout rate for LSTM layers
    #'dropout_rate_dense': [0, 0.2, 0.4, 0.6],  # Dropout rate for dense layers
    'num_lstm_layers': [1, 2, 3]
}

# Build the LSTM model
def build_model(units, num_nodes_per_layer, learning_rate, dropout_rate, dropout_rate_lstm, num_lstm_layers):
    model = Sequential()
    
    # Input LSTM layer
    model.add(LSTM(units=units, return_sequences=True, input_shape=X_sequences.shape[1:]))
    
    # Dropout layer after the input LSTM if dropout_rate > 0
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    
    # Additional LSTM layers
    for _ in range(num_lstm_layers - 1):
        model.add(LSTM(units=units, return_sequences=True))
        # Dropout layer after each additional LSTM layer if dropout_rate_lstm > 0
        if dropout_rate_lstm > 0:
            model.add(Dropout(dropout_rate_lstm))
      
    # Additional dropout layer before the Flatten and dense layers
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    
    # Flatten and dense layers
    model.add(Flatten())
    for nodes in num_nodes_per_layer:
        model.add(Dense(units=nodes, activation='relu'))
    
    # Output layer
    model.add(Dense(7))
    
    model.compile(
        loss=MeanSquaredError(),
        optimizer=Adam(learning_rate=learning_rate),
        metrics=[RootMeanSquaredError()]
    )

    return model
# Initialize the TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=3)

# Initialize the KerasRegressor with build_model function
keras_regressor = KerasRegressor(build_fn=build_model)

# Create a GridSearchCV object
grid_search = GridSearchCV(
    keras_regressor,
    param_grid,
    cv=tscv,
    n_jobs=-2,
    verbose=2
)

# Fit the GridSearchCV on your data
grid_search.fit(X_sequences, y_sequences)

# Get the best parameters from the grid search
best_params = grid_search.best_params_

# Build the final model using the best parameters
final_best_model = build_model(**best_params)

# Train the final best model on the entire dataset
final_best_history = final_best_model.fit(
    X_sequences, y_sequences,
    epochs=50,
    batch_size=32
)
# Plot average training and validation loss
plt.plot(final_best_history.history['loss'], label='Training Loss')
plt.plot(final_best_history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot average R-squared scores
plt.plot(validation_r2_scores, label='Validation R-squared')
plt.xlabel('Fold')
plt.ylabel('R-squared Score')
plt.legend()
plt.show()

# Calculate and print average R-squared score
avg_r2_score = np.mean(validation_r2_scores)
print(f"Average Validation R-squared Score: {avg_r2_score}")

# Plot predicted vs. actual values
plt.figure(figsize=(10, 6))
for i in range(len(X_sequences)):
    y_pred = final_best_model.predict(X_sequences[i:i+1])[0]
    plt.plot(y_pred, label=f'Fold {i+1} Prediction', linestyle='dashed')
    plt.plot(y_sequences[i], label=f'Fold {i+1} Actual')
plt.xlabel('Day')
plt.ylabel('Temperature')
plt.title('Predicted vs. Actual Temperature')
plt.legend()
plt.show()

# Evaluate the final model on the entire dataset
y_pred = final_best_model.predict(X_sequences)
final_mse = mean_squared_error(y_sequences, y_pred)
final_r2 = r2_score(y_sequences, y_pred)

print(f"Final Mean Squared Error: {final_mse}")
print(f"Final R-squared Score: {final_r2}")