
# -*- coding: utf-8 -*-

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

# Load your dataset
data = pd.read_csv(r'C:\Users\white\Desktop\Dissertation\data\temp_weather.csv')

# Define the sequence length (n-1 days)
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

"""
iterates through data, takes a slice from i to i+7 for X (input)
another slide from i+7 to i+14 for ouput
.values used to convert df slice into array
"""

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

# Define a parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}


# Create a Random Forest Regressor
xg_regressor = XGBRegressor(random_state=42)

# Create a TimeSeriesSplit object
tscv = TimeSeriesSplit(n_splits=5)


from sklearn.metrics import make_scorer

# Define a custom scoring function using per-sequence RMSE
def custom_rmse_scorer(y_true, y_pred):
    rmse_per_sequence = np.sqrt(np.mean((y_pred - y_true)**2, axis=1))
    return -np.mean(rmse_per_sequence)  # Use negative to convert minimizing to maximizing

"""
in scikit-lean the scoring functions are designed to be maximised, higher values indicate better performance
"""

# Convert the custom RMSE scorer into a scorer object
custom_scorer = make_scorer(custom_rmse_scorer)

# hyperparameter tuning
grid_search = GridSearchCV(estimator=xg_regressor, param_grid=param_grid, cv=tscv, n_jobs=-1, verbose=2, scoring=custom_scorer)


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
from sklearn.metrics import mean_absolute_error
mae= mean_absolute_error(y_test, y_pred_scaled)

# Calculate the R-squared score
r2 = r2_score(y_test, y_pred_scaled)
mape=mean_absolute_percentage_error(y_test,y_pred_scaled)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared Score: {r2}")
print(f"MAE: {mae}")


from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import matplotlib.dates as mdates
from datetime import datetime
from datetime import timedelta
from sklearn.metrics import mean_absolute_error

# Determine the number of data points corresponding to the months
num_months = 12
num_data_points = 30 * num_months  # Assuming an average of 30 data points per month


# Calculate the starting index for the subset
start_index = len(y_test) - num_data_points
start_index_1 = len(data) - num_data_points

# Extract the actual and predicted values for the subset
subset_y_test = y_test[start_index:]
subset_y_pred = y_pred_scaled[start_index:] 


subset_dates = data.iloc[start_index_1: start_index_1 + num_data_points]['date']
subset_dates


# Define a list of colors for the predicted day lines
predicted_colors = ['blue', 'green', 'red', 'purple', 'yellow', 'cyan', 'magenta']

# Plotting the predicted and real values for the subset
plt.figure(figsize=(12, 6))

# Plot the shifted actual temperature values as a single line
plt.plot(subset_dates, subset_y_test[:,0 ], label='Real Temperature', color='orange')

# Plot the predicted values for each day as dashed lines with different colors
for i in range(subset_y_test.shape[1]):
    plt.plot(subset_dates, subset_y_pred[:, i], label=f'Predicted Day {i+1}', color=predicted_colors[i], linestyle='--', alpha=0.5)

"""
loops through each days predicted temperature within the array
"""

plt.title('Predicted vs. Real Temperature (Latest 12 Months)')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.grid(True)
plt.legend()
plt.tight_layout()
from matplotlib.ticker import MaxNLocator

# Set custom x-axis ticks for evenly spaced dates
plt.xticks(rotation=45)  # Rotate tick labels for better visibility
locator = MaxNLocator(nbins=3)  # Adjust the number of desired ticks
plt.gca().xaxis.set_major_locator(locator)

plt.show()



# Calculate z-scores for the 'temperature' column
z_scores = (data['temp'] - data['temp'].mean()) / data['temp'].std()

# Set a threshold for outlier detection (e.g., 3 standard deviations)
threshold = 3

# Identify outliers
outliers_indices = np.where(np.abs(z_scores) > threshold)[0]
outliers_data = data.loc[outliers_indices]

# Create a DataFrame to display outliers
outliers_df = pd.DataFrame({'Temperature': outliers_data['temp'], 'Z-Score': z_scores[outliers_indices]})
print(outliers_df)

###TEST###
# Initialize lists to store evaluation metrics for each fold

mae_scores = []
rmse_scores = []
r2_scores = []

# Iterate through each fold in the cross-validation
for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled)):
    print(f"Fold {fold_idx + 1}:")
    print(train_idx)
    print(val_idx)
    # Split data into train and validation sets for this fold
    X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    # Fit the model with the grid search on the fold's training data
    grid_search.fit(X_fold_train, y_fold_train)

    # Get the best model from the grid search
    best_model = grid_search.best_estimator_

    # Make predictions on the validation data
    y_pred_val = best_model.predict(X_fold_val)


    
    # Calculate MAE for this fold
    fold_mae = mean_absolute_error(y_fold_val, y_pred_val)
    mae_scores.append(fold_mae)

    # Calculate RMSE for this fold
    fold_rmse = sqrt(mean_squared_error(y_fold_val, y_pred_val))
    rmse_scores.append(fold_rmse)

    # Calculate R-squared for this fold
    fold_r2 = r2_score(y_fold_val, y_pred_val)
    r2_scores.append(fold_r2)

    
    print(f"Fold {fold_idx + 1} MAE: {fold_mae}")
    print(f"Fold {fold_idx + 1} RMSE: {fold_rmse}")
    print(f"Fold {fold_idx + 1} R-squared: {fold_r2}")

# Create a figure for the graphs
plt.figure(figsize=(10, 6))



# Plot RMSE scores for each fold
plt.subplot(222)
plt.plot(range(1, len(rmse_scores) + 1), rmse_scores, marker='o', color='r')
plt.title('RMSE Scores for Each Fold')
plt.xlabel('Fold')
plt.ylabel('RMSE')

# Plot R-squared scores for each fold
plt.subplot(223)
plt.plot(range(1, len(r2_scores) + 1), r2_scores, marker='o', color='g')
plt.title('R-squared Scores for Each Fold')
plt.xlabel('Fold')
plt.ylabel('R-squared')

# Plot MAE scores for each fold
plt.subplot(224)
plt.plot(range(1, len(mae_scores) + 1), mae_scores, marker='o', color='g')
plt.title('MAE Scores for Each Fold')
plt.xlabel('Fold')
plt.ylabel('MAE')

# Adjust layout and display the plots
plt.tight_layout()
plt.show()

# Predict temperature values using the best model
y_pred_scaled = best_model.predict(X_test_scaled)

# Reshape the predicted values to match the original shape of y_test
y_pred = y_pred_scaled.reshape(y_test.shape)

# Create a figure for the graph
plt.figure(figsize=(12, 6))

# Plot the actual temperature values
plt.plot(y_test, label='Actual Temperature', color='blue')

# Plot the predicted temperature values
plt.plot(y_pred, label='Predicted Temperature', color='orange')

plt.title('Actual vs. Predicted Temperature Values')
plt.xlabel('Time')
plt.ylabel('Temperature')

# Create separate legends for actual and predicted temperature
legend_actual = plt.legend(handles=[plt.Line2D([0], [0], color='blue')], labels=['Actual Temperature'])
legend_predicted = plt.legend(handles=[plt.Line2D([0], [0], color='orange')],  labels=['Predicted Temperature'])
# Combine handles and labels from both legends
all_handles = legend_actual.legendHandles + legend_predicted.legendHandles
all_labels = [label.get_text() for label in legend_actual.get_texts()] + [label.get_text() for label in legend_predicted.get_texts()]

# Create a single legend with combined handles and labels
plt.legend(all_handles, all_labels, loc="upper left")

plt.grid(True)
plt.tight_layout()
plt.show()









