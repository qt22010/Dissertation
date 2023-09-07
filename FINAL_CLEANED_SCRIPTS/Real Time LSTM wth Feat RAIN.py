"""
@author: white
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError

# Load dataset
pre_data = pd.read_csv(r'C:\Users\white\Desktop\Dissertation\data\rain_weather_real_time.csv')

# Load donor dataset

donor = pd.read_csv(r'C:\Users\white\Desktop\Dissertation\data\donor_precipitation.csv')

if not donor.empty:
    pre_data=pre_data.drop(pre_data.index[0])
    # Concatenate donor below pre_data
    data = pd.concat([pre_data, donor], axis=0, ignore_index=True)
else:
    # Handle the case when the donor DataFrame is empty
    print(f"The file donor.csv is empty.") 
    data = pre_data

# Define the sequence length (n-1 days)
sequence_length = 7

features = ['dew_point', 'wind_speed', 'pressure', 'humidity', 'clouds_all', 'rain_1h', 'day_of_week', 'month', 'year', 'season']
target_temp = 'rain_1h'  # Target variable for temperature prediction

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

# Extract and use the last sequence from X_scaled and y_sequences
X_real = X_sequences[-1]
y_real = y_sequences[-1]

# Remove the last sequence from X_scaled and y_sequences for testing purposes
X_scaled = X_sequences[:-1]
y_sequences = y_sequences[:-1]

# Define the hyperparameters
units = 32
num_nodes_per_layer = [32, 64]
learning_rate = 0.01
dropout_rate = 0.2
num_lstm_layers = 1

# Build the LSTM model
def build_model(units, learning_rate, dropout_rate, num_lstm_layers, num_nodes_per_layer):
    model = Sequential()
    
    # Input LSTM layer
    model.add(LSTM(units=units, return_sequences=True, input_shape=(sequence_length, len(features))))
    
    # Dropout layer after the input LSTM if dropout_rate > 0
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    
    # Additional LSTM layers
    for _ in range(num_lstm_layers - 1):
        model.add(LSTM(units=units, return_sequences=True))
        # Dropout layer after each additional LSTM layer if dropout_rate > 0
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
      
    # Additional dropout layer before the Flatten and dense layers
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    
    # Flatten and dense layers (with constant number of nodes)
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

# Create the model
retrained_model = build_model(units, learning_rate, dropout_rate, num_lstm_layers, num_nodes_per_layer)

# Train the model with the new data
retrained_model.fit(X_scaled, y_sequences, epochs=50, batch_size=32, verbose=1)

X_real = X_real.reshape(-1,7,10)
week_predictions = retrained_model.predict(X_real)

print(week_predictions)




