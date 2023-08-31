
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load your historical weather data and preprocess it
data = pd.read_csv(r'C:\Users\white\Desktop\Dissertation\data\weather_data_cleaned1.csv')

# Select relevant features for prediction
features = ['dew_point', 'feels_like', 'temp_min', 'temp_max', 'pressure', 'humidity', 'wind_speed', 'wind_deg', 'clouds_all', 'rain_1h', 'snow_1h']
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

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load your historical weather data and preprocess it
data = pd.read_csv(r'C:\Users\white\Desktop\Dissertation\data\weather_data_cleaned1.csv')

# Select relevant features for prediction
features = ['snow_1h', 'dew_point', 'feels_like', 'temp_min', 'temp_max', 'pressure', 'humidity', 'wind_speed', 'wind_deg', 'clouds_all']
target_rain = 'rain_1h'  # Target variable for rain prediction

# Splitting the data into features and targets
X = data[features]
y = data[target_rain]  # Only predict rain for now

# Feature scaling using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Number of days to forecast ahead
forecast_horizon = 7

# List to store predicted rain for the next 7 days
predicted_rains = []

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

# Predicting rain for the next 7 days
for i, model in enumerate(models):
    predicted_rain = model.predict(current_input)[0]
    predicted_rains.append(predicted_rain)
    
    # Update current input for the next prediction
    current_input = current_input.copy()
    current_input[0, -1] = predicted_rain  # Replace the last feature (rain) with the prediction

# Print the predicted rain for the next 7 days
for day, rain in enumerate(predicted_rains, start=1):
    print(f'Predicted rain for day {day}: {rain:.2f}')

# Actual rain data (replace with actual values)
actual_rain_data = [2.21, 0, 11.39, 5.01, 0, 0.26, 10.16]

# Convert the predicted_rains list into a list of lists
predicted_rain_data = [[rain] for rain in predicted_rains]

# Flatten the predicted_rain_data list
predicted_rain_flat = [rain[0] for rain in predicted_rain_data]

mae_scores = [mean_absolute_error(actual_rain_data, predicted_rain_flat)]
rmse_scores = [mean_squared_error(actual_rain_data, predicted_rain_flat, squared=False)]

# Print MAE and RMSE for each day
for day, (actual_rain, predicted_rain) in enumerate(zip(actual_rain_data, predicted_rain_data), start=1):
    mae = mean_absolute_error([actual_rain], [predicted_rain[0]])
    rmse = mean_squared_error([actual_rain], [predicted_rain[0]], squared=False)
    
    print(f'Day {day}:')
    print(f'Actual rain: {actual_rain:.2f}')
    print(f'Predicted rain: {predicted_rain[0]:.2f}')
    print(f'MAE: {mae:.2f}')
    print(f'RMSE: {rmse:.2f}')
    print('---')

# Create a single figure for the line plot
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

# Line plot: Actual vs. Predicted Rain
legend_lines = []  # To store lines for legend
orange_x_coords = []
orange_y_coords = []
blue_x_coords = []
blue_y_coords = []

for day, (actual_rains, predicted_rain) in enumerate(zip(actual_rain_data, predicted_rains), start=1):
    actual_line = axes.plot([day], [actual_rains], marker='o', markersize=8, color='blue', label='Actual Rain')
    predicted_line = axes.plot([day], [predicted_rain], marker='x', markersize=8, color='orange', linestyle='dashed', label='Predicted Rain')
    
    # Add lines to the legend
    if day == 1:
        legend_lines.extend([actual_line[0], predicted_line[0]])
        # Connect points with lines
    axes.plot([day, day], [actual_rains, predicted_rain], color='red', linestyle='-', alpha=0.5)
    # Accumulate coordinates for orange and blue points
    orange_x_coords.append(day)
    orange_y_coords.append(predicted_rain)
    blue_x_coords.append(day)
    blue_y_coords.append(actual_rains)

# Draw lines through all orange and blue points
axes.plot(orange_x_coords, orange_y_coords, color='orange', linestyle='-', alpha=0.5)
axes.plot(blue_x_coords, blue_y_coords, color='blue', linestyle='-', alpha=0.5)

axes.set_xlabel('Day')
axes.set_ylabel('Rain')
axes.set_title('Actual vs. Predicted Rain')
axes.legend(legend_lines, ['Actual Rain', 'Predicted Rain'])

# Adjust layout and show the plot
plt.tight_layout()
plt.show()


