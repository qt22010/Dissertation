# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load your merged dataset
merged_dataset = pd.read_csv('merged_dataset.csv')  # Replace with your dataset file

# Convert date column to datetime format
merged_dataset['Date1'] = pd.to_datetime(merged_dataset['Date1'])

# Define the features and target variable
features = ['WindSpeed', 'WindDirection', 'O3', 'PM2.5', 'PM10']
target = 'PollenCount'

# Create lag features for sequences
sequence_length = 7  # Number of past days to consider for prediction
X_sequences = []
y_sequences = []

for i in range(len(merged_dataset) - sequence_length - 7):
    X_sequences.append(merged_dataset[features].iloc[i:i+sequence_length])
    y_sequences.append(merged_dataset[target].iloc[i+sequence_length+7])  # Predict 7 days ahead

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10],
    'epsilon': [0.01, 0.1, 0.2, 0.5]
}

grid_search = GridSearchCV(SVR(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

best_svm_model = grid_search.best_estimator_

# Visualization of Hyperparameter Tuning Results
results = grid_search.cv_results_

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(results['param_C'], results['mean_test_score'], marker='o')
plt.title('Hyperparameter Tuning: C')
plt.xlabel('C')
plt.ylabel('Mean Test Score')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(range(len(results['param_kernel'])), results['mean_test_score'], marker='o')
plt.title('Hyperparameter Tuning: Kernel')
plt.xlabel('Kernel')
plt.ylabel('Mean Test Score')
plt.xticks(range(len(results['param_kernel'])), results['param_kernel'], rotation=45)
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(results['param_gamma'], results['mean_test_score'], marker='o')
plt.title('Hyperparameter Tuning: Gamma')
plt.xlabel('Gamma')
plt.ylabel('Mean Test Score')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(results['param_epsilon'], results['mean_test_score'], marker='o')
plt.title('Hyperparameter Tuning: Epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Mean Test Score')
plt.grid(True)

plt.tight_layout()
plt.show()

# Make predictions for the next 7 days using the best model
predictions = best_svm_model.predict(X_test_scaled)

# Evaluate the model
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)
f1 = f1_score(y_test, np.where(predictions > 0.5, 1, 0), average='macro')  # Assuming binary classification

print("R-squared (R2):", r2)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("F1 Score:", f1)