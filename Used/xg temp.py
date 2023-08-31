
# -*- coding: utf-8 -*-

print("start custom metric temp rf")
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
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]



# Create a Random Forest Regressor
xg_regressor = XGBRegressor(random_state=42)

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
grid_search = GridSearchCV(estimator=xg_regressor, param_grid=param_grid, cv=tscv, n_jobs=-1, verbose=2, scoring=custom_scorer)


# Fit the model with the grid search
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Use the best model to make predictions
best_model = grid_search.best_estimator_

y_pred_scaled = best_model.predict(X_test_scaled)



#day 1
new_data1=pd.read_csv(r'C:\Users\white\Desktop\Dissertation\data\temp_weather_real_time_30.csv')


# Extract specific columns for features
new_data_features = new_data1[features]
new_data_features

# Convert the features to a NumPy array
new_data_array = new_data_features.to_numpy()
last_week_data = new_data_features.iloc[-13:-6]

# Convert the extracted data to a NumPy array
last_week_data_array = last_week_data.to_numpy()

# Reshape the last week's data for scaling (1, sequence_length * num_features)
last_week_data_reshaped = last_week_data_array.reshape(1, sequence_length * len(features))

# Calculate mean and standard deviation from the training data
mean = X_train_scaled.mean(axis=0)
std = X_train_scaled.std(axis=0)

# Manually scale the last week's data using mean and standard deviation
last_week_data_scaled = (last_week_data_reshaped - mean) / std

# Reshape the scaled data for model input (1, sequence_length, num_features)
last_week_sequence_final = last_week_data_scaled.reshape(1, sequence_length, len(features))

# Make predictions using the trained model
predicted_temperatures_scaled = best_model.predict(last_week_data_scaled )


# Print the predicted temperatures for the next week
print("Predicted Temperatures for Next Week:", predicted_temperatures_scaled)




#day 2
new_data2=pd.read_csv(r'C:\Users\white\Desktop\Dissertation\data\temp_weather_real_time_31.csv')
# Extract specific columns for features
new_data_features = new_data2[features]
new_data_features

# Convert the features to a NumPy array
new_data_array = new_data_features.to_numpy()
last_week_data = new_data_features.iloc[-12:-5]

# Convert the extracted data to a NumPy array
last_week_data_array = last_week_data.to_numpy()

# Reshape the last week's data for scaling (1, sequence_length * num_features)
last_week_data_reshaped = last_week_data_array.reshape(1, sequence_length * len(features))

# Calculate mean and standard deviation from the training data
mean = X_train_scaled.mean(axis=0)
std = X_train_scaled.std(axis=0)

# Manually scale the last week's data using mean and standard deviation
last_week_data_scaled = (last_week_data_reshaped - mean) / std

# Reshape the scaled data for model input (1, sequence_length, num_features)
last_week_sequence_final = last_week_data_scaled.reshape(1, sequence_length, len(features))

# Make predictions using the trained model
predicted_temperatures_scaled = best_model.predict(last_week_data_scaled )


# Print the predicted temperatures for the next week
print("Predicted Temperatures for Next Week:", predicted_temperatures_scaled)

#day 3
new_data3=pd.read_csv(r'C:\Users\white\Desktop\Dissertation\data\temp_weather_real_time_01.csv')
# Extract specific columns for features
new_data_features = new_data3[features]
new_data_features

# Convert the features to a NumPy array
new_data_array = new_data_features.to_numpy()
last_week_data = new_data_features.iloc[-11:-4]

# Convert the extracted data to a NumPy array
last_week_data_array = last_week_data.to_numpy()

# Reshape the last week's data for scaling (1, sequence_length * num_features)
last_week_data_reshaped = last_week_data_array.reshape(1, sequence_length * len(features))

# Calculate mean and standard deviation from the training data
mean = X_train_scaled.mean(axis=0)
std = X_train_scaled.std(axis=0)

# Manually scale the last week's data using mean and standard deviation
last_week_data_scaled = (last_week_data_reshaped - mean) / std

# Reshape the scaled data for model input (1, sequence_length, num_features)
last_week_sequence_final = last_week_data_scaled.reshape(1, sequence_length, len(features))

# Make predictions using the trained model
predicted_temperatures_scaled = best_model.predict(last_week_data_scaled )


# Print the predicted temperatures for the next week
print("Predicted Temperatures for Next Week:", predicted_temperatures_scaled)


#day 4
new_data4=pd.read_csv(r'C:\Users\white\Desktop\Dissertation\data\temp_weather_real_time_02.csv')
# Extract specific columns for features
new_data_features = new_data4[features]
new_data_features

# Convert the features to a NumPy array
new_data_array = new_data_features.to_numpy()
last_week_data = new_data_features.iloc[-10:-3]

# Convert the extracted data to a NumPy array
last_week_data_array = last_week_data.to_numpy()

# Reshape the last week's data for scaling (1, sequence_length * num_features)
last_week_data_reshaped = last_week_data_array.reshape(1, sequence_length * len(features))

# Calculate mean and standard deviation from the training data
mean = X_train_scaled.mean(axis=0)
std = X_train_scaled.std(axis=0)

# Manually scale the last week's data using mean and standard deviation
last_week_data_scaled = (last_week_data_reshaped - mean) / std

# Reshape the scaled data for model input (1, sequence_length, num_features)
last_week_sequence_final = last_week_data_scaled.reshape(1, sequence_length, len(features))

# Make predictions using the trained model
predicted_temperatures_scaled = best_model.predict(last_week_data_scaled )


# Print the predicted temperatures for the next week
print("Predicted Temperatures for Next Week:", predicted_temperatures_scaled)


#day 5
new_data5=pd.read_csv(r'C:\Users\white\Desktop\Dissertation\data\temp_weather_real_time_03.csv')
# Extract specific columns for features
new_data_features = new_data5[features]
new_data_features

# Convert the features to a NumPy array
new_data_array = new_data_features.to_numpy()
last_week_data = new_data_features.iloc[-9:-2]

# Convert the extracted data to a NumPy array
last_week_data_array = last_week_data.to_numpy()

# Reshape the last week's data for scaling (1, sequence_length * num_features)
last_week_data_reshaped = last_week_data_array.reshape(1, sequence_length * len(features))

# Calculate mean and standard deviation from the training data
mean = X_train_scaled.mean(axis=0)
std = X_train_scaled.std(axis=0)

# Manually scale the last week's data using mean and standard deviation
last_week_data_scaled = (last_week_data_reshaped - mean) / std

# Reshape the scaled data for model input (1, sequence_length, num_features)
last_week_sequence_final = last_week_data_scaled.reshape(1, sequence_length, len(features))

# Make predictions using the trained model
predicted_temperatures_scaled = best_model.predict(last_week_data_scaled )


# Print the predicted temperatures for the next week
print("Predicted Temperatures for Next Week:", predicted_temperatures_scaled)


#day 6
new_data6=pd.read_csv(r'C:\Users\white\Desktop\Dissertation\data\temp_weather_real_time_04.csv')
# Extract specific columns for features
new_data_features = new_data6[features]
new_data_features

# Convert the features to a NumPy array
new_data_array = new_data_features.to_numpy()
last_week_data = new_data_features.iloc[-8:-1]

# Convert the extracted data to a NumPy array
last_week_data_array = last_week_data.to_numpy()

# Reshape the last week's data for scaling (1, sequence_length * num_features)
last_week_data_reshaped = last_week_data_array.reshape(1, sequence_length * len(features))

# Calculate mean and standard deviation from the training data
mean = X_train_scaled.mean(axis=0)
std = X_train_scaled.std(axis=0)

# Manually scale the last week's data using mean and standard deviation
last_week_data_scaled = (last_week_data_reshaped - mean) / std

# Reshape the scaled data for model input (1, sequence_length, num_features)
last_week_sequence_final = last_week_data_scaled.reshape(1, sequence_length, len(features))

# Make predictions using the trained model
predicted_temperatures_scaled = best_model.predict(last_week_data_scaled )


# Print the predicted temperatures for the next week
print("Predicted Temperatures for Next Week:", predicted_temperatures_scaled)


#day 7

new_data7=pd.read_csv(r'C:\Users\white\Desktop\Dissertation\data\temp_weather_real_time_05.csv')
# Extract specific columns for features
new_data_features = new_data7[features]
new_data_features

# Convert the features to a NumPy array
new_data_array = new_data_features.to_numpy()
last_week_data = new_data_features.iloc[-7:]

# Convert the extracted data to a NumPy array
last_week_data_array = last_week_data.to_numpy()

# Reshape the last week's data for scaling (1, sequence_length * num_features)
last_week_data_reshaped = last_week_data_array.reshape(1, sequence_length * len(features))

# Calculate mean and standard deviation from the training data
mean = X_train_scaled.mean(axis=0)
std = X_train_scaled.std(axis=0)

# Manually scale the last week's data using mean and standard deviation
last_week_data_scaled = (last_week_data_reshaped - mean) / std

# Reshape the scaled data for model input (1, sequence_length, num_features)
last_week_sequence_final = last_week_data_scaled.reshape(1, sequence_length, len(features))

# Make predictions using the trained model
predicted_temperatures_scaled = best_model.predict(last_week_data_scaled )


# Print the predicted temperatures for the next week
print("Predicted Temperatures for Next Week:", predicted_temperatures_scaled)



import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

dates = ['31/07/2023', '01/08/2023', '02/08/2023', '03/08/2023', '04/08/2023', '05/08/2023', '06/08/2023', '07/08/2023', '08/08/2023', '09/08/2023', '10/08/2023', '11/08/2023', '12/08/2023']
values = [17.83125, 18.02208333, 17.05583333, 17.81291667, 16.96916667, 14.80192308, 15.5425, 16.90166667, 15.33153846, 18.58833333, 21.12375, 21.06625, 18.8225]
additional_values_1 =[21.381905, 19.364904, 16.722008 ,14.536601 ,13.874912, 13.399724, 12.31158 ]
additional_values_2 =[21.447226, 19.364904, 16.722008, 14.536601, 13.874912, 13.399724, 12.31158 ]
additional_values_3 =[21.447226, 19.364904 ,16.722008, 14.536601 ,13.874912, 13.399724, 12.31158 ]
additional_values_4 =[21.447226, 19.364904, 16.722008, 14.536601 ,14.068617, 13.399724 ,12.31158 ]
additional_values_5 =[21.447226, 19.364904, 16.722008, 14.577294 ,13.874912, 13.399724 ,12.31158 ]
additional_values_6 =[21.447226, 19.364904,16.862131 ,14.536601, 13.874912, 13.399724 ,12.31158 ]
additional_values_7 =[21.414085, 19.307228, 16.722008, 14.536601, 13.874912, 13.399724 ,12.31158 ]


# Convert date strings to datetime objects
date_objects = [datetime.strptime(date, '%d/%m/%Y') for date in dates]

plt.figure(figsize=(10, 6))

# Create the line plot for the original data
plt.plot(date_objects, values, marker='o', label='Actual Temperature')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Real-time temperature')

# Create lines connecting the additional values to their corresponding dates for each scenario
plt.plot(date_objects[:7], additional_values_1, marker='o', color='r', linestyle='--', label='Forecast 1')
plt.plot(date_objects[1:8], additional_values_2, marker='o', color='b', linestyle='--', label='Forecast 2')
plt.plot(date_objects[2:9], additional_values_3, marker='o', color='g', linestyle='--', label='Forecast 3')
plt.plot(date_objects[3:10], additional_values_4, marker='o', color='m', linestyle='--', label='Forecast 4')
plt.plot(date_objects[4:11], additional_values_5, marker='o', color='c', linestyle='--', label='Forecast 5')
plt.plot(date_objects[5:12], additional_values_5, marker='o', color='y', linestyle='--', label='Forecast 6')
plt.plot(date_objects[6:13], additional_values_5, marker='o', color='k', linestyle='--', label='Forecast 7')

# Show legend
plt.legend()

plt.tight_layout()
plt.show()










################
# Iterate through the sequences
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Assuming X_sequences contains your initial 7 sequences and y_sequences contains corresponding targets
# Also, assuming you have a new_data_point that you want to incorporate in real-time

# Create an array to hold predicted temperature values
predicted_temperatures = []

# Create an array to hold the corresponding dates
predicted_dates = []

# Iterate through the sequences
for sequence_index in range(len(X_sequences)):
    new_sequence = X_sequences[sequence_index:sequence_index + 1]  # Get the new sequence for prediction
    new_sequence_scaled = scaler.transform(new_sequence.reshape(1, -1))  # Scale the new sequence
    
    # Update model and predict
    best_model.fit(X_train_scaled, y_train)  # You might want to consider using accumulated_data_scaled for retraining
    predicted_next_7_days = best_model.predict(new_sequence_scaled)
    
    # Incorporate the actual next day's data point into the sequence
    new_data_point_scaled = scaler.transform(new_data_point.reshape(1, -1))  # Scale the new data point
    updated_sequence_scaled = np.concatenate((new_sequence_scaled[:, 1:], new_data_point_scaled), axis=1)
    
    # Update X_sequences and y_sequences with the updated sequence
    X_sequences[sequence_index] = updated_sequence_scaled
    y_sequences[sequence_index] = new_data_point_scaled[:, -1]
    
    # Append the predicted temperature and date
    predicted_temperatures.append(predicted_next_7_days[0])
    predicted_dates.append(data.index[sequence_index + sequence_length])

# Plotting the graph
actual_temperatures = data[target_temp].values[sequence_length:]

plt.figure(figsize=(10, 6))
plt.plot(predicted_dates, predicted_temperatures, label='Predicted Temperatures')
plt.plot(predicted_dates, actual_temperatures, label='Actual Temperatures')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Predicted vs Actual Temperatures')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()






#########
###########













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

# Determine the number of days to shift the actual temperature values
shift_days = 1

# Shift the actual temperature y values to the right by 7 days
subset_y_test_shifted = subset_y_test[shift_days:]
subset_y_test_dates_shifted = subset_dates[shift_days:]

# Define a list of colors for the predicted day lines
predicted_colors = ['blue', 'green', 'red', 'purple', 'yellow', 'cyan', 'magenta']

# Plotting the predicted and real values for the subset
plt.figure(figsize=(12, 6))

# Plot the shifted actual temperature values as a single line
plt.plot(subset_dates, subset_y_test[:,0 ], label='Real Temperature', color='orange')

# Plot the predicted values for each day as dashed lines with different colors
for i in range(subset_y_test.shape[1]):
    plt.plot(subset_dates, subset_y_pred[:, i], label=f'Predicted Day {i+1}', color=predicted_colors[i], linestyle='--', alpha=0.5)

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

# Calculate R-squared for each sequence
r2_per_sequence = []
for i in range(len(y_test)):
    r2_per_sequence.append(r2_score(y_test[i], y_pred_scaled[i]))

# Convert the list to a numpy array for consistency
r2_per_sequence = np.array(r2_per_sequence)

average_r2 = np.mean(r2_per_sequence)

print("Average R-squared:", average_r2)

print("end custom metric temp")

# Generate example data (replace this with your dataset)



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









