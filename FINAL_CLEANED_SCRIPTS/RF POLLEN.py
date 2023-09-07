

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

# Load your dataset
data = pd.read_csv(r'C:\Users\white\Desktop\Dissertation\data\pollen\final_pollen_weather.csv')

# Define the sequence length (n-1 days)
sequence_length = 7

features = ['Poac', 'temp', 'humidity', 'wind_speed', 'clouds_all', 'dew_point','day_of_week',	'month',	'year',	'season']

target_temp = ['Poac']  # Target variable for temperature prediction


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

num_samples = y_sequences.shape[0]
y_sequences = y_sequences.reshape(num_samples, -1)


# Calculate the index where the split should occur
split_index = int(len(data) * 0.8)  # 80% train, 20% test

# Splitting into training and testing sets
X_train, X_test = X_sequences[:split_index], X_sequences[split_index:]
y_train, y_test = y_sequences[:split_index], y_sequences[split_index:]

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the scaler on the training data and transform the training and test data
X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0], -1))

print(y_test.shape)
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
tscv = TimeSeriesSplit(n_splits=5)
print(tscv)

from sklearn.metrics import make_scorer

# Define a custom scoring function using per-sequence RMSE
def custom_rmse_scorer(y_true, y_pred):
    rmse_per_sequence = np.sqrt(np.mean((y_pred - y_true)**2, axis=1))
    return -np.mean(rmse_per_sequence)  # Use negative to convert minimizing to maximizing

# Convert the custom RMSE scorer into a scorer object
custom_scorer = make_scorer(custom_rmse_scorer)

# ... (rest of your code for hyperparameter tuning)
grid_search = GridSearchCV(estimator=rf_regressor , param_grid=param_grid, cv=tscv, n_jobs=-2, verbose=2, scoring=custom_scorer)


# Fit the model with the grid search
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Use the best model to make predictions
best_model = grid_search.best_estimator_

y_pred_scaled = best_model.predict(X_test_scaled)

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred_scaled)
rmse = sqrt(mse)

# Calculate the R-squared score
r2 = r2_score(y_test, y_pred_scaled)
mape=mean_absolute_percentage_error(y_test,y_pred_scaled)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared Score: {r2}")


from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import matplotlib.dates as mdates
from datetime import datetime
from datetime import timedelta
from sklearn.metrics import mean_absolute_error


# Determine the number of data points corresponding to the months
num_months = 50
num_data_points = 30 * num_months  # Assuming an average of 30 data points per month

# Calculate the starting index for the subset
start_index = len(y_test) - num_data_points

# Extract the actual and predicted values for the subset
subset_y_test = y_test[start_index:]
subset_y_pred = y_pred_scaled[start_index:]

# Define a list of colors for the predicted day lines
predicted_colors = ['blue', 'green', 'red', 'purple', 'yellow', 'cyan', 'magenta']

# Plotting the predicted and real values for the subset
plt.figure(figsize=(12, 6))

# Plot the shifted actual temperature values as a single line
plt.plot(subset_y_test[:, 0], label='Real Temperature', color='orange')

# Plot the predicted values for each day as dashed lines with different colors
for i in range(subset_y_pred.shape[1]):
    plt.plot(subset_y_pred[:, i], label=f'Predicted Day {i+1}', color=predicted_colors[i], linestyle='--', alpha=0.5)

plt.title('Predicted vs. Real Temperature Values Temp Custom metric all input cust metric (Latest 12 Months)')
plt.xlabel('Data Point Index')
plt.ylabel('Temperature')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Calculate MAE and RMSE for each sequence
mae_per_sequence = np.mean(np.abs(y_pred_scaled - y_test), axis=1)
rmse_per_sequence = np.sqrt(np.mean((y_pred_scaled - y_test)**2, axis=1))


# Calculate overall average of MAE and RMSE
average_mae = np.mean(mae_per_sequence)
average_rmse = np.mean(rmse_per_sequence)

print("Average MAE:", average_mae)
print("Average RMSE:", average_rmse)

