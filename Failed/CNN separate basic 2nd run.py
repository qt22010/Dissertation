import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# Load and preprocess data
data = pd.read_csv('weather_data.csv')
X = data[['mean_temp', 'max_temp', 'min_temp', 'pressure', 'clouds', 'dew_point', 'wind_speed', 'wind_direction', 'humidity', 'snow']]
y_temp = data['mean_temp']
y_precip = data['rain']

# Define the look-ahead period (1 week)
look_ahead = 7

# Create lag features for sequences
X_sequences = []
y_sequences_temp = []
y_sequences_precip = []

for i in range(len(X) - look_ahead):
    X_sequences.append(X.iloc[i:i+look_ahead].values)
    y_sequences_temp.append(y_temp.iloc[i+look_ahead])
    y_sequences_precip.append(y_precip.iloc[i+look_ahead])

X_sequences = np.array(X_sequences)
y_sequences_temp = np.array(y_sequences_temp)
y_sequences_precip = np.array(y_sequences_precip)

# Split data into train and test sets
X_train, X_test, y_temp_train, y_temp_test, y_precip_train, y_precip_test = train_test_split(
    X_sequences, y_sequences_temp, y_sequences_precip, test_size=0.1, random_state=42, shuffle=False
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[2]))
X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[2]))

X_train_scaled = X_train_scaled.reshape(X_train.shape)
X_test_scaled = X_test_scaled.reshape(X_test.shape)

# Build CNN models
model_temp = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(look_ahead, X.shape[1])),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1)  # For temperature regression
])

model_precip = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(look_ahead, X.shape[1])),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # For precipitation classification
])

# Compile the models
model_temp.compile(optimizer='adam', loss='mean_squared_error')
model_precip.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the models for temperature and precipitation prediction
history_temp = model_temp.fit(X_train_scaled, y_temp_train, epochs=50, batch_size=32, verbose=0)
history_precip = model_precip.fit(X_train_scaled, y_precip_train, epochs=50, batch_size=32, verbose=0)

# Visualize Training Loss for Temperature model
plt.plot(history_temp.history['loss'], label='Temperature Model Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Temperature Model Training Loss')
plt.legend()
plt.show()

# Visualize Training Loss for Precipitation model
plt.plot(history_precip.history['loss'], label='Precipitation Model Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Precipitation Model Training Loss')
plt.legend()
plt.show()

# Predictions for Temperature
predictions_temp = model_temp.predict(X_test_scaled).flatten()

# Predictions for Precipitation
predictions_precip_prob = model_precip.predict(X_test_scaled).flatten()
predictions_precip = np.where(predictions_precip_prob > 0.5, 1, 0)

# Evaluate metrics for Temperature
mse_temp = mean_squared_error(y_temp_test, predictions_temp)
rmse_temp = np.sqrt(mse_temp)
mae_temp = mean_absolute_error(y_temp_test, predictions_temp)
r2_temp = r2_score(y_temp_test, predictions_temp)
f1_score_temp = f1_score(y_temp_test, np.round(predictions_temp))

# Evaluate metrics for Precipitation
accuracy_precip = np.mean(predictions_precip == y_precip_test)
f1_score_precip = f1_score(y_precip_test, predictions_precip)
mse_precip = mean_squared_error(y_precip_test, predictions_precip_prob)
rmse_precip = np.sqrt(mse_precip)
mae_precip = mean_absolute_error(y_precip_test, predictions_precip_prob)
r2_precip = r2_score(y_precip_test, predictions_precip_prob)


# Create plots for Temperature predictions
plt.figure(figsize=(10, 6))
plt.plot(y_temp_test, label='Actual Temperature', marker='o')
plt.plot(predictions_temp, label='Predicted Temperature', marker='o')
plt.xlabel('Days')
plt.ylabel('Temperature')
plt.title('Actual vs. Predicted Temperature')
plt.legend()
plt.show()

# Create plots for Precipitation predictions
plt.figure(figsize=(10, 6))
plt.plot(y_precip_test, label='Actual Precipitation', marker='o')
plt.plot(predictions_precip_prob, label='Predicted Precipitation', marker='o')
plt.xlabel('Days')
plt.ylabel('Precipitation Probability')
plt.title('Actual vs. Predicted Precipitation Probability')
plt.legend()
plt.show()

# Print evaluation metrics for Temperature
print("Metrics for Temperature Prediction (One Week Ahead):")
print(f'R-squared (R2): {r2_temp:.2f}')
print(f'Mean Squared Error (MSE): {mse_temp:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse_temp:.2f}')
print(f'Mean Absolute Error (MAE): {mae_temp:.2f}')
print(f'F1 Score: {f1_score_temp:.2f}')

# Print evaluation metrics for Precipitation
print("\nMetrics for Precipitation Prediction (One Week Ahead):")
print(f'Accuracy: {accuracy_precip:.2f}')
print(f'F1 Score: {f1_score_precip:.2f}')
print(f'Mean Squared Error (MSE): {mse_precip:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse_precip:.2f}')
print(f'Mean Absolute Error (MAE): {mae_precip:.2f}')
print(f'R-squared (R2): {r2_precip:.2f}')