import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score

# Load and preprocess data
data = pd.read_csv('weather_data.csv')
X = data[['mean_temp', 'max_temp', 'min_temp', 'pressure', 'clouds', 'dew_point', 'wind_speed', 'wind_direction', 'humidity', 'snow']]
y_temp = data['mean_temp']
y_precip = data['rain']

# Split data into train and test sets
X_train, X_test, y_temp_train, y_temp_test, y_precip_train, y_precip_test = train_test_split(
    X, y_temp, y_precip, test_size=0.1, random_state=42, shuffle=False
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define parameter grid for MLP hyperparameter tuning
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01]
}

# Grid search for Temperature prediction
grid_search_temp = GridSearchCV(MLPRegressor(random_state=42), param_grid, cv=5)
grid_search_temp.fit(X_train_scaled, y_temp_train)
best_model_temp = grid_search_temp.best_estimator_

# Grid search for Precipitation prediction
grid_search_precip = GridSearchCV(MLPRegressor(random_state=42), param_grid, cv=5)
grid_search_precip.fit(X_train_scaled, y_precip_train)
best_model_precip = grid_search_precip.best_estimator_

# Multi-step predictions for Temperature
look_ahead = 7
predictions_temp = []

for i in range(len(X_test_scaled) - look_ahead + 1):
    input_data = X_test_scaled[i:i+1]
    prediction = best_model_temp.predict(input_data)
    predictions_temp.append(prediction[0])

# Multi-step predictions for Precipitation
predictions_precip = []

for i in range(len(X_test_scaled) - look_ahead + 1):
    input_data = X_test_scaled[i:i+1]
    prediction = best_model_precip.predict(input_data)
    predictions_precip.append(prediction[0])

# Evaluate metrics for Temperature
mse_temp = mean_squared_error(y_temp_test[look_ahead-1:], predictions_temp)
rmse_temp = np.sqrt(mse_temp)
mae_temp = mean_absolute_error(y_temp_test[look_ahead-1:], predictions_temp)
r2_temp = r2_score(y_temp_test[look_ahead-1:], predictions_temp)

# Evaluate metrics for Precipitation
mse_precip = mean_squared_error(y_precip_test[look_ahead-1:], predictions_precip)
rmse_precip = np.sqrt(mse_precip)
mae_precip = mean_absolute_error(y_precip_test[look_ahead-1:], predictions_precip)
r2_precip = r2_score(y_precip_test[look_ahead-1:], predictions_precip)

# Visualize predictions for Temperature
plt.plot(y_temp_test.index[look_ahead-1:], y_temp_test.values[look_ahead-1:], label='Actual Temperature')
plt.plot(y_temp_test.index[look_ahead-1:], predictions_temp, label='Predicted Temperature', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Multi-Step MLP Predicted Temperature')
plt.legend()
plt.show()

# Visualize predictions for Precipitation
plt.plot(y_precip_test.index[look_ahead-1:], y_precip_test.values[look_ahead-1:], label='Actual Precipitation')
plt.plot(y_precip_test.index[look_ahead-1:], predictions_precip, label='Predicted Precipitation', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Precipitation')
plt.title('Multi-Step MLP Predicted Precipitation')
plt.legend()
plt.show()

# Print evaluation metrics for Temperature
print("Multi-Step MLP Metrics for Temperature:")
print(f'Mean Squared Error (MSE): {mse_temp:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse_temp:.2f}')
print(f'Mean Absolute Error (MAE): {mae_temp:.2f}')
print(f'R-squared (R2): {r2_temp:.2f}')

# Print evaluation metrics for Precipitation
print("Multi-Step MLP Metrics for Precipitation:")
print(f'Mean Squared Error (MSE): {mse_precip:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse_precip:.2f}')
print(f'Mean Absolute Error (MAE): {mae_precip:.2f}')
print(f'R-squared (R2): {r2_precip:.2f}')