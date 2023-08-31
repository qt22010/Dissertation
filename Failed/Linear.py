# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

# Load and preprocess data
data = pd.read_csv('weather_data.csv')
X = data[['mean_temp', 'max_temp', 'min_temp', 'pressure', 'clouds', 'dew_point', 'wind_speed', 'wind_direction', 'humidity', 'snow']]
y_precipitation = data['rain']
y_mean_temperature = data['mean_temp']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_precip_train, y_precip_test, y_temp_train, y_temp_test = train_test_split(
    X_scaled, y_precipitation, y_mean_temperature, test_size=0.2, random_state=42, shuffle=False
)

# Baseline Linear Regression model for mean temperature
baseline_model_temp = LinearRegression()
baseline_model_temp.fit(X_train, y_temp_train)

# Predictions
temp_pred_baseline = baseline_model_temp.predict(X_test)
temp_mse_baseline = mean_squared_error(y_temp_test, temp_pred_baseline)

print("Baseline Linear Regression for Mean Temperature:")
print(f'Mean Temperature MSE: {temp_mse_baseline:.2f}')

# Baseline Linear Regression model
baseline_model = LinearRegression()
baseline_model.fit(X_train, y_precip_train)  # For precipitation

# Predictions
precip_pred_baseline = baseline_model.predict(X_test)
precip_mse_baseline = mean_squared_error(y_precip_test, precip_pred_baseline)

print("Baseline Linear Regression:")
print(f'Precipitation MSE: {precip_mse_baseline:.2f}')

# Visualization for Baseline Linear Regression
plt.figure(figsize=(12, 6))

# Scatter plot: Actual vs. Predicted (Precipitation)
plt.subplot(1, 2, 1)
plt.scatter(y_precip_test, precip_pred_baseline, alpha=0.5)
plt.title('Baseline Linear Regression: Precipitation Prediction')
plt.xlabel('Actual Precipitation')
plt.ylabel('Predicted Precipitation')

# Scatter plot: Actual vs. Predicted (Mean Temperature)
plt.subplot(1, 2, 2)
plt.scatter(y_temp_test, temp_pred_baseline, alpha=0.5)
plt.title('Baseline Linear Regression: Mean Temperature Prediction')
plt.xlabel('Actual Mean Temperature')
plt.ylabel('Predicted Mean Temperature')

plt.tight_layout()
plt.show()

# Hyperparameter tuning using GridSearchCV

# Linear Regression
linear_params = {'fit_intercept': [True, False]}
linear_grid = GridSearchCV(LinearRegression(), linear_params, cv=5)
linear_grid.fit(X_train, y_precip_train)  # For precipitation
best_linear_model_precip = linear_grid.best_estimator_

linear_grid.fit(X_train, y_temp_train)  # For mean temperature
best_linear_model_temp = linear_grid.best_estimator_

# Polynomial Regression
poly_params = {'degree': [1, 2, 3]}
poly_grid = GridSearchCV(PolynomialFeatures(), poly_params, cv=5)
poly_grid.fit(X_train)
best_poly_degree = poly_grid.best_params_['degree']

poly_features = PolynomialFeatures(degree=best_poly_degree)
X_poly_train = poly_features.fit_transform(X_train)
X_poly_test = poly_features.transform(X_test)

linear_grid.fit(X_poly_train, y_precip_train)  # For precipitation with polynomial features
best_poly_model_precip = linear_grid.best_estimator_

linear_grid.fit(X_poly_train, y_temp_train)  # For mean temperature with polynomial features
best_poly_model_temp = linear_grid.best_estimator_

# Ridge Regression
ridge_params = {'alpha': [0.01, 0.1, 1, 10]}
ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=5)
ridge_grid.fit(X_train, y_precip_train)  # For precipitation
best_ridge_model_precip = ridge_grid.best_estimator_

ridge_grid.fit(X_train, y_temp_train)  # For mean temperature
best_ridge_model_temp = ridge_grid.best_estimator_

# Lasso Regression
lasso_params = {'alpha': [0.01, 0.1, 1, 10]}
lasso_grid = GridSearchCV(Lasso(), lasso_params, cv=5)
lasso_grid.fit(X_train, y_precip_train)  # For precipitation
best_lasso_model_precip = lasso_grid.best_estimator_

lasso_grid.fit(X_train, y_temp_train)  # For mean temperature
best_lasso_model_temp = lasso_grid.best_estimator_

# Model evaluation
precip_pred_linear = best_linear_model_precip.predict(X_test)
temp_pred_linear = best_linear_model_temp.predict(X_test)

precip_pred_poly = best_poly_model_precip.predict(X_poly_test)
temp_pred_poly = best_poly_model_temp.predict(X_poly_test)

precip_pred_ridge = best_ridge_model_precip.predict(X_test)
temp_pred_ridge = best_ridge_model_temp.predict(X_test)

precip_pred_lasso = best_lasso_model_precip.predict(X_test)
temp_pred_lasso = best_lasso_model_temp.predict(X_test)

precip_mse_linear = mean_squared_error(y_precip_test, precip_pred_linear)
temp_mse_linear = mean_squared_error(y_temp_test, temp_pred_linear)

precip_mse_poly = mean_squared_error(y_precip_test, precip_pred_poly)
temp_mse_poly = mean_squared_error(y_temp_test, temp_pred_poly)

precip_mse_ridge = mean_squared_error(y_precip_test, precip_pred_ridge)
temp_mse_ridge = mean_squared_error(y_temp_test, temp_pred_ridge)

precip_mse_lasso = mean_squared_error(y_precip_test, precip_pred_lasso)
temp_mse_lasso = mean_squared_error(y_temp_test, temp_pred_lasso)

# Visualization
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.scatter(y_precip_test, precip_pred_linear, alpha=0.5)
plt.title('Linear Regression: Precipitation Prediction')
plt.xlabel('Actual Precipitation')
plt.ylabel('Predicted Precipitation')

plt.subplot(2, 2, 2)
plt.scatter(y_temp_test, temp_pred_linear, alpha=0.5)
plt.title('Linear Regression: Mean Temperature Prediction')
plt.xlabel('Actual Mean Temperature')
plt.ylabel('Predicted Mean Temperature')

plt.subplot(2, 2, 3)
plt.scatter(y_precip_test, precip_pred_poly, alpha=0.5)
plt.title('Polynomial Regression: Precipitation Prediction')
plt.xlabel('Actual Precipitation')
plt.ylabel('Predicted Precipitation')

plt.subplot(2, 2, 4)
plt.scatter(y_temp_test, temp_pred_poly, alpha=0.5)
plt.title('Polynomial Regression: Mean Temperature Prediction')
plt.xlabel('Actual Mean Temperature')
plt.ylabel('Predicted Mean Temperature')

plt.tight_layout()
plt.show()

print("Linear Regression:")
print(f'Precipitation MSE: {precip_mse_linear:.2f}')
print(f'Mean Temperature MSE: {temp_mse_linear:.2f}')

print("\nPolynomial Regression:")
print(f'Precipitation MSE: {precip_mse_poly:.2f}')
print(f'Mean Temperature MSE: {temp_mse_poly:.2f}')

print("\nRidge Regression:")
print(f'Precipitation MSE: {precip_mse_ridge:.2f}')
print(f'Mean Temperature MSE: {temp_mse_ridge:.2f}')

print("\nLasso Regression:")
print(f'Precipitation MSE: {precip_mse_lasso:.2f}')
print(f'Mean Temperature MSE: {temp_mse_lasso:.2f}')