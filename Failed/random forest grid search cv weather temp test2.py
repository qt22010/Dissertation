
#################################
####################################
############################
################################
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load and preprocess your historical weather data
data = pd.read_csv(r'C:\Users\white\Desktop\Dissertation\data\temp_weather.csv')

# Select relevant features for prediction
features = ['temp','dew_point', 'temp_min', 'temp_max', 'humidity', 'clouds_all', 'snow_1h']
target_temp = 'temp'  # Target variable for temperature prediction

# Splitting the data into features and targets
X = data[features]
y = data[target_temp]  # Only predict temperature for now

# Feature scaling using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Number of days to forecast ahead
forecast_horizon = 7

# List to store predicted temperatures for the next 7 days
predicted_temperatures = []

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=forecast_horizon, shuffle=False
)

# Create a list of shifts for each day in the forecast horizon
shifts = list(range(forecast_horizon))

# Creating a list of Random Forest models, one for each day in the forecast horizon
models = [RandomForestRegressor(n_estimators=100, random_state=42) for _ in range(forecast_horizon)]

# Training each model for its respective day
for i, model in enumerate(models):
    # Shift the target to predict the i-th day ahead
    shifted_y_train = y_train.shift(-shifts[i]).dropna()
    model.fit(X_train[:len(shifted_y_train)], shifted_y_train)

# Initial input for prediction
current_input = X_test[0].reshape(1, -1)

# Predicting temperatures for the next 7 days
for i, model in enumerate(models):
    predicted_temperature = model.predict(current_input)[0]
    predicted_temperatures.append(predicted_temperature)
    
    # Update current input for the next prediction
    current_input = current_input.copy()
    current_input[0, 0] = predicted_temperature  # Replace the first feature (temperature) with the prediction

# Print the predicted temperatures for the next 7 days
for day, temp in enumerate(predicted_temperatures, start=1):
    print(f'Predicted temperature for day {day}: {temp:.2f}')

# Actual temperature data (replace with actual values)
actual_temperature_data = [15.77, 16.43, 16.55, 19.19, 19.36, 19.17, 17.38]

# Convert the predicted_temperatures list into a list of lists
predicted_temperature_data = [[temp] for temp in predicted_temperatures]

# Flatten the predicted_temperature_data list
predicted_temperature_flat = [temp[0] for temp in predicted_temperature_data]

mae_scores = [mean_absolute_error(actual_temperature_data, predicted_temperature_flat)]
rmse_scores = [mean_squared_error(actual_temperature_data, predicted_temperature_flat, squared=False)]


# Print MAE and RMSE for each day
# Print MAE and RMSE for each day
for day, (actual_temp, predicted_temp) in enumerate(zip(actual_temperature_data, predicted_temperature_data), start=1):
    mae = mean_absolute_error([actual_temp], [predicted_temp[0]])
    rmse = mean_squared_error([actual_temp], [predicted_temp[0]], squared=False)
    
    print(f'Day {day}:')
    print(f'Actual temperature: {actual_temp:.2f}')
    print(f'Predicted temperature: {predicted_temp[0]:.2f}')
    print(f'MAE: {mae:.2f}')
    print(f'RMSE: {rmse:.2f}')
    print('---')


# Create a single figure for the line plot
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

# Line plot: Actual vs. Predicted Temperatures
legend_lines = []  # To store lines for legend
orange_x_coords = []
orange_y_coords = []
blue_x_coords = []
blue_y_coords = []

for day, (actual_temps, predicted_temp) in enumerate(zip(actual_temperature_data, predicted_temperatures), start=1):
    actual_line = axes.plot([day], [actual_temps], marker='o', markersize=8, color='blue', label='Actual')
    predicted_line = axes.plot([day], [predicted_temp], marker='x', markersize=8, color='orange', linestyle='dashed', label='Predicted')
    
    # Add lines to the legend
    if day == 1:
        legend_lines.extend([actual_line[0], predicted_line[0]])
        # Connect points with lines
    axes.plot([day, day], [actual_temps, predicted_temp], color='red', linestyle='-', alpha=0.5)
    # Accumulate coordinates for orange and blue points
    orange_x_coords.append(day)
    orange_y_coords.append(predicted_temp)
    blue_x_coords.append(day)
    blue_y_coords.append(actual_temps)

# Draw lines through all orange and blue points
axes.plot(orange_x_coords, orange_y_coords, color='orange', linestyle='-', alpha=0.5)
axes.plot(blue_x_coords, blue_y_coords, color='blue', linestyle='-', alpha=0.5)

axes.set_xlabel('Day')
axes.set_ylabel('Temperature')
axes.set_title('Actual vs. Predicted Temperatures')
axes.legend(legend_lines, ['Actual', 'Predicted'])

# Adjust layout and show the plot
plt.tight_layout()
plt.show()


import numpy as np
from sklearn.metrics import r2_score

# Load your dataset
data = pd.read_csv(r'C:\Users\white\Desktop\Dissertation\data\temp_weather.csv')

# Define the sequence length (n-1 days)
sequence_length = 7


features = ['dew_point', 'temp_min', 'temp_max', 'humidity', 'clouds_all', 'snow_1h','day_of_week','month','year','season']
target_temp = 'temp'  # Target variable for temperature prediction

# Splitting the data into features and targets
X = data[features]
y = data[target_temp]  # Only predict temperature for now


# Create sequences and corresponding target values
X_sequences = []
y_sequences = []

for i in range(len(data) - sequence_length - 7):
    X_sequences.append(X.iloc[i:i+sequence_length].values)  # Slice n-1 days of features
    y_sequences.append(y.iloc[i+sequence_length:i+sequence_length+7].values)  # Next 7 days of temperatures

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)

# Reshape the input sequences
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

# Create the Random Forest Regressor (or any other suitable model)
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
rf_regressor.fit(X_train_reshaped, y_train)

# Make predictions
y_pred = rf_regressor.predict(X_test_reshaped)

# Evaluate the model (e.g., calculate the mean squared error for all predicted days)

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
from math import sqrt
# Calculate the Root Mean Squared Error (RMSE)
rmse = sqrt(mse)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
# Calculate the R-squared score
r2 = r2_score(y_test, y_pred)
print(f"R-squared Score: {r2}")


# Determine the number of data points corresponding to the latest 6 months
num_months = 6
num_data_points = 30 * num_months  # Assuming an average of 30 data points per month

# Calculate the starting index for the subset
start_index = len(y_test) - num_data_points

# Extract the actual and predicted values for the subset
subset_y_test = y_test[start_index:]
subset_y_pred = y_pred[start_index:]

# Plotting the predicted and real values for the subset
plt.figure(figsize=(12, 6))
plt.plot(subset_y_test, label='Real', color='orange')
plt.plot(subset_y_pred, label='Predicted', color='blue', linestyle='--')

plt.title('Predicted vs. Real Temperature Values (Latest 6 Months)')
plt.xlabel('Data Point Index')
plt.ylabel('Temperature')
plt.grid(True)
plt.legend()
plt.show()
#####################test start
# Create an array of days for plotting (1 to 7 days)
# Define the date range you're interested in (replace with your desired range)
data['date'] = pd.to_datetime(data['date'])
# Define the date range you're interested in (replace with your desired range)
start_date = '2020-11-01'
end_date = '2022-11-01'

# Convert the date range to datetime objects
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Filter the data based on the date range
filtered_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

# Get the corresponding indices for the filtered data
filtered_indices = filtered_data.index

# Filter the prediction arrays based on the indices
filtered_y_pred = y_pred[filtered_indices]
filtered_y_test = y_test[filtered_indices]

# Plotting the predicted and real values within the specified date range
plt.figure(figsize=(12, 6))
plt.plot(filtered_data['date'], filtered_y_pred, label='Predicted', color='blue')
plt.plot(filtered_data['date'], filtered_y_test, label='Real', color='orange')

plt.title('Predicted vs. Real Temperature Values (Date Range)')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.grid(True)
plt.legend()
plt.show()
################test end
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
predicted_temperatures = rf_regressor.predict(recent_features_flattened)

print("Predicted Temperatures for the Next 7 Days:")
print(predicted_temperatures)


# Predicted temperatures
predicted_temperatures = np.array([[16.77612774, 16.97409733, 17.21288868, 17.17135833, 17.4129405, 17.75097914, 17.93978197]])

# Actual temperatures
actual_temperatures = np.array([16.42708333, 16.5528, 19.19333333, 19.36375, 19.17, 17.38166667, 17.83125])

# Calculate the squared differences between predicted and actual temperatures
squared_diff = (predicted_temperatures - actual_temperatures)**2

# Calculate the mean squared difference
mean_squared_diff = np.mean(squared_diff)

# Calculate the Root Mean Squared Error (RMSE)
rmse = sqrt(mean_squared_diff)

print(f"Root Mean Squared Error: {rmse}")

#########################################
#########################################
#########################################

#hypertuning:


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
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

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)

# Create a Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=42)

# Define a parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=3, n_jobs=-2, verbose=2)

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

#################
###################

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
y_test
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

# Determine the number of data points corresponding to the latest 6 months
num_months = 6
num_data_points = 30 * num_months  # Assuming an average of 30 data points per month

# Calculate the starting index for the subset
start_index = len(y_test) - num_data_points

# Extract the actual and predicted values for the subset
subset_y_test = y_test[start_index:]
subset_y_pred = y_pred_scaled[start_index:]

# Plotting the predicted and real values for the subset
plt.figure(figsize=(12, 6))
plt.plot(subset_y_test, label='Real', color='orange')
plt.plot(subset_y_pred, label='Predicted', color='blue', linestyle='--')

plt.title('Predicted vs. Real Temperature Values 1111111(Latest 6 Months)')
plt.xlabel('Data Point Index')
plt.ylabel('Temperature')
plt.grid(True)
plt.legend()
plt.show()
print(y_test)


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