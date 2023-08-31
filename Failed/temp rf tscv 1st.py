# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from math import sqrt

# Load your dataset
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

# Calculate the index where the split should occur
split_index = int(len(data) * 0.8)  # 80% train, 20% test

# Splitting into training and testing sets
X_train, X_test = X_sequences[:split_index], X_sequences[split_index:]
y_train, y_test = y_sequences[:split_index], y_sequences[split_index:]

# Define a parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=42)

# Create a TimeSeriesSplit object
tscv = TimeSeriesSplit(n_splits=3)

# Create a GridSearchCV object with time series cross-validation
grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=tscv, n_jobs=-2, verbose=2)

# Fit the model with the grid search
grid_search.fit(X_train.reshape(X_train.shape[0], -1), y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Use the best model to make predictions
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test.reshape(X_test.shape[0], -1))

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
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from math import sqrt

# Load your dataset
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

# Define a parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a TimeSeriesSplit object
tscv = TimeSeriesSplit(n_splits=3)

# Create a Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=42)

# Create a GridSearchCV object with time series cross-validation
grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=tscv, n_jobs=-2, verbose=2)

# Fit the model with the grid search
grid_search.fit(X_sequences.reshape(X_sequences.shape[0], -1), y_sequences)

# Get the best parameters from the grid search
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Use the best model to make predictions using the entire dataset
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_sequences.reshape(X_sequences.shape[0], -1))

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_sequences, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_sequences, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared Score: {r2}")

# Sample data for the next 7 days 
recent_features = [
    [10.05708333, 11.69, 21.21, 67.95833333, 63.45833333, 0, 1, 7, 2023, 3],  # 25th/07/23
    [11.098, 8.08, 23.95, 72.76, 60.6, 0, 2, 7, 2023, 3],  # 26th/07/23
    [17.01875, 15.32, 22.83, 87.5, 86.33333333, 0, 3, 7, 2023, 3],  # 27th/07/23
    [15.41833333, 15.88, 24.05, 78.41666667, 86.41666667, 0, 4, 7, 2023, 3],  # 28th/07/23
    [13.36833333, 15.58, 23.99, 70.45833333, 43.66666667, 0, 5, 7, 2023, 3],  # 29th/07/23
    [13.77083333, 13.94, 21.7, 80.04166667, 66.70833333, 0, 6, 7, 2023, 3],  # 30th/07/23
    [15.15291667, 14.49, 21.27, 84.625, 70.79166667, 0, 0, 7, 2023, 3]  # 31st/07/23
    # ... continue with data for the remaining days
]

# Create a DataFrame for the next 7 days' features
recent_features_flattened = np.array(recent_features).reshape(1, -1)

# Predict the temperatures for the next 7 days
predicted_temperatures = best_model.predict(recent_features_flattened)

print("Predicted Temperatures for the Next 7 Days:")
print(predicted_temperatures)
