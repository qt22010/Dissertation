# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score
import xgboost as xgb
# Load and preprocess data
data = pd.read_csv('weather_data.csv')
X = data[['mean_temp', 'max_temp', 'min_temp', 'pressure', 'clouds', 'dew_point', 'wind_speed', 'wind_direction', 'humidity', 'snow']]
y_precipitation = data['rain']
y_mean_temperature = data['mean_temp']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create lag features for sequences
sequence_length = 7  # Number of past days to consider for prediction
X_sequences = []
y_sequences_precip = []
y_sequences_temp = []

for i in range(len(X_scaled) - sequence_length):
    X_sequences.append(X_scaled[i:i+sequence_length])
    y_sequences_precip.append(y_precipitation[i+sequence_length])
    y_sequences_temp.append(y_mean_temperature[i+sequence_length])

X_sequences = np.array(X_sequences)
y_sequences_precip = np.array(y_sequences_precip)
y_sequences_temp = np.array(y_sequences_temp)

# Train-test split
X_train, X_test, y_precip_train, y_precip_test, y_temp_train, y_temp_test = train_test_split(
    X_sequences, y_sequences_precip, y_sequences_temp, test_size=0.1, random_state=42, shuffle=False
)
# Hyperparameter tuning using GridSearchCV for XGBoost Regression
param_grid_xgb = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search_xgb_precip = GridSearchCV(xgb.XGBRegressor(random_state=42), param_grid_xgb, cv=5)
grid_search_xgb_precip.fit(X_train, y_precip_train)

grid_search_xgb_temp = GridSearchCV(xgb.XGBRegressor(random_state=42), param_grid_xgb, cv=5)
grid_search_xgb_temp.fit(X_train, y_temp_train)

best_xgb_model_precip = grid_search_xgb_precip.best_estimator_
best_xgb_model_temp = grid_search_xgb_temp.best_estimator_

# Visualization of XGBoost Regression hyperparameter tuning results
results_xgb_precip = grid_search_xgb_precip.cv_results_
results_xgb_temp = grid_search_xgb_temp.cv_results_

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(results_xgb_precip['mean_test_score'], marker='o')
plt.title('Hyperparameter Tuning (XGBoost): Precipitation Prediction')
plt.xlabel('Hyperparameter Set')
plt.ylabel('Mean Test Score')
plt.xticks(range(len(results_xgb_precip['params'])), range(len(results_xgb_precip['params'])))
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(results_xgb_temp['mean_test_score'], marker='o')
plt.title('Hyperparameter Tuning (XGBoost): Mean Temperature Prediction')
plt.xlabel('Hyperparameter Set')
plt.ylabel('Mean Test Score')
plt.xticks(range(len(results_xgb_temp['params'])), range(len(results_xgb_temp['params'])))
plt.grid(True)

plt.tight_layout()
plt.show()

# Predict for the upcoming week using XGBoost Regression models
week_weather_pred_xgb_temp = best_xgb_model_temp.predict(X_test)
week_precip_pred_xgb = best_xgb_model_precip.predict(X_test)

# Evaluation metrics
def evaluate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return r2, mse, rmse, mae

def evaluate_f1(y_true, y_pred):
    y_pred_class = np.where(y_pred > 0.5, 1, 0)  # Convert to binary classes
    f1 = f1_score(y_true, y_pred_class)
    return f1
# Evaluate XGBoost Regression (Mean Temperature)
r2_xgb_temp, mse_xgb_temp, rmse_xgb_temp, mae_xgb_temp = evaluate_metrics(y_temp_test, week_weather_pred_xgb_temp)
f1_xgb_temp = evaluate_f1(y_temp_test, week_weather_pred_xgb_temp)

# Evaluate XGBoost Regression (Precipitation)
r2_xgb_precip, mse_xgb_precip, rmse_xgb_precip, mae_xgb_precip = evaluate_metrics(y_precip_test, week_precip_pred_xgb)
f1_xgb_precip = evaluate_f1(y_precip_test, week_precip_pred_xgb)

# Print evaluation metrics for XGBoost Regression
print("XGBoost Regression Metrics (Mean Temperature):")
print(f'R-squared (R2): {r2_xgb_temp:.2f}')
print(f'Mean Squared Error (MSE): {mse_xgb_temp:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse_xgb_temp:.2f}')
print(f'Mean Absolute Error (MAE): {mae_xgb_temp:.2f}')
print(f'F1 Score: {f1_xgb_temp:.2f}')

print("XGBoost Regression Metrics (Precipitation):")
print(f'R-squared (R2): {r2_xgb_precip:.2f}')
print(f'Mean Squared Error (MSE): {mse_xgb_precip:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse_xgb_precip:.2f}')
print(f'Mean Absolute Error (MAE): {mae_xgb_precip:.2f}')
print(f'F1 Score: {f1_xgb_precip:.2f}')