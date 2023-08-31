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

# Load and preprocess data
data = pd.read_csv(r'C:\Users\white\Desktop\Dissertation\data\weather_data_cleaned.csv')

X = data[['dew_point',
'feels_like',
'temp_min',
'temp_max',
'pressure',
'humidity',
'wind_speed',
'wind_deg',
'snow_1h',
'clouds_all']]
y_precipitation = data['rain_1h']   # Combined precipitation from both columns
y_mean_temperature = data['temp']

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

print(y_sequences_temp)

# Train-test split
X_train, X_test, y_precip_train, y_precip_test, y_temp_train, y_temp_test = train_test_split(
    X_sequences, y_sequences_precip, y_sequences_temp, test_size=0.1, random_state=42, shuffle=False
)

# Reshape sequences for Linear Regression
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

# Train Linear Regression model
model_precip = LinearRegression()
model_temp = LinearRegression()

model_precip.fit(X_train_reshaped, y_precip_train)
model_temp.fit(X_train_reshaped, y_temp_train)

# Predict for the upcoming week
week_weather_pred = model_temp.predict(X_test_reshaped)
week_precip_pred = model_precip.predict(X_test_reshaped)

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

# Evaluate Linear Regression (Mean Temperature)
r2_linear_temp, mse_linear_temp, rmse_linear_temp, mae_linear_temp = evaluate_metrics(y_temp_test, week_weather_pred)


# Evaluate Linear Regression (Precipitation)
r2_linear_precip, mse_linear_precip, rmse_linear_precip, mae_linear_precip = evaluate_metrics(y_precip_test, week_precip_pred)


# Print evaluation metrics
print("Linear Regression Metrics (Mean Temperature):")
print(f'R-squared (R2): {r2_linear_temp:.2f}')
print(f'Mean Squared Error (MSE): {mse_linear_temp:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse_linear_temp:.2f}')
print(f'Mean Absolute Error (MAE): {mae_linear_temp:.2f}')


print("Linear Regression Metrics (Precipitation):")
print(f'R-squared (R2): {r2_linear_precip:.2f}')
print(f'Mean Squared Error (MSE): {mse_linear_precip:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse_linear_precip:.2f}')
print(f'Mean Absolute Error (MAE): {mae_linear_precip:.2f}')


# Visualization
days = range(len(y_temp_test))
plt.plot(days, y_temp_test, label='Actual Mean Temp', marker='o')
plt.plot(days, week_weather_pred, label='Predicted Mean Temp', marker='o')
plt.xlabel('Days')
plt.ylabel('Mean Temperature')
plt.title('Predicted Mean Temperature for Upcoming Week')
plt.legend()
plt.show()

plt.plot(days, y_precip_test, label='Actual Precipitation', marker='o')
plt.plot(days, week_precip_pred, label='Predicted Precipitation', marker='o')
plt.xlabel('Days')
plt.ylabel('Precipitation')
plt.title('Predicted Precipitation for Upcoming Week')
plt.legend()
plt.show()

# Hyperparameter tuning using GridSearchCV for Ridge Regression
param_grid_ridge = {
    'alpha': [0.01, 0.1, 1, 10, 100]
}

grid_search_ridge_precip = GridSearchCV(Ridge(), param_grid_ridge, cv=5)
grid_search_ridge_precip.fit(X_train, y_precip_train)

grid_search_ridge_temp = GridSearchCV(Ridge(), param_grid_ridge, cv=5)
grid_search_ridge_temp.fit(X_train, y_temp_train)

best_ridge_model_precip = grid_search_ridge_precip.best_estimator_
best_ridge_model_temp = grid_search_ridge_temp.best_estimator_

# Hyperparameter tuning using GridSearchCV for Lasso Regression
param_grid_lasso = {
    'alpha': [0.01, 0.1, 1, 10, 100]
}

grid_search_lasso_precip = GridSearchCV(Lasso(), param_grid_lasso, cv=5)
grid_search_lasso_precip.fit(X_train, y_precip_train)

grid_search_lasso_temp = GridSearchCV(Lasso(), param_grid_lasso, cv=5)
grid_search_lasso_temp.fit(X_train, y_temp_train)

best_lasso_model_precip = grid_search_lasso_precip.best_estimator_
best_lasso_model_temp = grid_search_lasso_temp.best_estimator_

# Visualization of Ridge Regression hyperparameter tuning results
results_ridge_precip = grid_search_ridge_precip.cv_results_
results_ridge_temp = grid_search_ridge_temp.cv_results_

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(param_grid_ridge['alpha'], results_ridge_precip['mean_test_score'], marker='o')
plt.title('Hyperparameter Tuning (Ridge): Precipitation Prediction')
plt.xlabel('Alpha')
plt.ylabel('Mean Test Score')

plt.subplot(1, 2, 2)
plt.plot(param_grid_ridge['alpha'], results_ridge_temp['mean_test_score'], marker='o')
plt.title('Hyperparameter Tuning (Ridge): Mean Temperature Prediction')
plt.xlabel('Alpha')
plt.ylabel('Mean Test Score')

plt.tight_layout()
plt.show()

# Visualization of Lasso Regression hyperparameter tuning results
results_lasso_precip = grid_search_lasso_precip.cv_results_
results_lasso_temp = grid_search_lasso_temp.cv_results_

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(param_grid_lasso['alpha'], results_lasso_precip['mean_test_score'], marker='o')
plt.title('Hyperparameter Tuning (Lasso): Precipitation Prediction')
plt.xlabel('Alpha')
plt.ylabel('Mean Test Score')

plt.subplot(1, 2, 2)
plt.plot(param_grid_lasso['alpha'], results_lasso_temp['mean_test_score'], marker='o')
plt.title('Hyperparameter Tuning (Lasso): Mean Temperature Prediction')
plt.xlabel('Alpha')
plt.ylabel('Mean Test Score')

plt.tight_layout()
plt.show()

# Predict for the upcoming week using Ridge Regression models
week_weather_pred_ridge_temp = best_ridge_model_temp.predict(X_test)
week_precip_pred_ridge = best_ridge_model_precip.predict(X_test)

# Predict for the upcoming week using Lasso Regression models
week_weather_pred_lasso_temp = best_lasso_model_temp.predict(X_test)
week_precip_pred_lasso = best_lasso_model_precip.predict(X_test)


# Evaluate Ridge Regression (Mean Temperature)
r2_ridge_temp, mse_ridge_temp, rmse_ridge_temp, mae_ridge_temp = evaluate_metrics(y_temp_test, week_weather_pred_ridge_temp)
f1_ridge_temp = evaluate_f1(y_temp_test, week_weather_pred_ridge_temp)

# Evaluate Ridge Regression (Precipitation)
r2_ridge_precip, mse_ridge_precip, rmse_ridge_precip, mae_ridge_precip = evaluate_metrics(y_precip_test, week_precip_pred_ridge)
f1_ridge_precip = evaluate_f1(y_precip_test, week_precip_pred_ridge)

# Evaluate Lasso Regression (Mean Temperature)
r2_lasso_temp, mse_lasso_temp, rmse_lasso_temp, mae_lasso_temp = evaluate_metrics(y_temp_test, week_weather_pred_lasso_temp)
f1_lasso_temp = evaluate_f1(y_temp_test, week_weather_pred_lasso_temp)

# Evaluate Lasso Regression (Precipitation)
r2_lasso_precip, mse_lasso_precip, rmse_lasso_precip, mae_lasso_precip = evaluate_metrics(y_precip_test, week_precip_pred_lasso)
f1_lasso_precip = evaluate_f1(y_precip_test, week_precip_pred_lasso)

# Print evaluation metrics
print("Ridge Regression Metrics (Mean Temperature):")
print(f'R-squared (R2): {r2_ridge_temp:.2f}')
print(f'Mean Squared Error (MSE): {mse_ridge_temp:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse_ridge_temp:.2f}')
print(f'Mean Absolute Error (MAE): {mae_ridge_temp:.2f}')
print(f'F1 Score: {f1_ridge_temp:.2f}')

print("Ridge Regression Metrics (Precipitation):")
print(f'R-squared (R2): {r2_ridge_precip:.2f}')
print(f'Mean Squared Error (MSE): {mse_ridge_precip:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse_ridge_precip:.2f}')
print(f'Mean Absolute Error (MAE): {mae_ridge_precip:.2f}')
print(f'F1 Score: {f1_ridge_precip:.2f}')

print("Lasso Regression Metrics (Mean Temperature):")
print(f'R-squared (R2): {r2_lasso_temp:.2f}')
print(f'Mean Squared Error (MSE): {mse_lasso_temp:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse_lasso_temp:.2f}')
print(f'Mean Absolute Error (MAE): {mae_lasso_temp:.2f}')
print(f'F1 Score: {f1_lasso_temp:.2f}')

print("Lasso Regression Metrics (Precipitation):")
print(f'R-squared (R2): {r2_lasso_precip:.2f}')
print(f'Mean Squared Error (MSE): {mse_lasso_precip:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse_lasso_precip:.2f}')
print(f'Mean Absolute Error (MAE): {mae_lasso_precip:.2f}')
print(f'F1 Score: {f1_lasso_precip:.2f}')