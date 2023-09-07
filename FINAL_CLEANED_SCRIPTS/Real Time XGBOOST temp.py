"""
@author: white
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

# Load dataset
pre_data = pd.read_csv(r'C:\Users\white\Desktop\Dissertation\data\temp_weather_real_time.csv')

# Load donor dataset

donor = pd.read_csv(r'C:\Users\white\Desktop\Dissertation\data\donor.csv')

if not donor.empty:
    pre_data=pre_data.drop(pre_data.index[0])
    # Concatenate donor below pre_data
    data = pd.concat([pre_data, donor], axis=0, ignore_index=True)
else:
    # Handle the case when the donor DataFrame is empty
    print(f"The file donor.csv is empty.") 
    data = pre_data

# Define the sequence length 
sequence_length = 7

features = ['temp','dew_point', 'temp_min', 'temp_max', 'humidity', 'clouds_all', 'snow_1h', 'day_of_week', 'month', 'year', 'season']
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

# Create StandardScaler instance
scaler = StandardScaler()

# Fit the scaler on all the data and transform it
X_scaled = scaler.fit_transform(X_sequences.reshape(X_sequences.shape[0], -1))

# Extract and use the last sequence from X_scaled and y_sequences
X_real = X_scaled[-1]
y_real = y_sequences[-1]

# Remove the last sequence from X_scaled and y_sequences for testing purposes
X_scaled = X_scaled[:-1]
y_sequences = y_sequences[:-1]


# Define the hyperparameters obtained from hyperparameter tuning
n_estimators = 50
max_depth = 3
learning_rate = 0.1
subsample =  1.0
colsample_bytree = 1.0 


# Create a new XGBoost with the hyperparameters from tscv
retrained_model = XGBRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    subsample=subsample,
    colsample_bytree=colsample_bytree
    
)

# Fit the newly created model with the new data
retrained_model.fit(X_scaled, y_sequences)  

# Use the retrained model to make predictions on new data
X_real = X_real.reshape(1, -1)
week_predictions = retrained_model.predict(X_real)

print(week_predictions)
