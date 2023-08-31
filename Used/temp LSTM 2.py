import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from numpy.random import seed
from tqdm import tqdm

seed(42)
tf.random.set_seed(42)
np.random.seed(42)


# Load your dataset and preprocess as needed
data = pd.read_csv('temp_weather.csv')

# Define the sequence length (n-1 days)
sequence_length = 7

features = ['temp','dew_point', 'temp_min', 'temp_max', 'humidity', 'clouds_all', 'snow_1h', 'day_of_week', 'month', 'year', 'season']
target_temp = 'temp'  # Target variable for temperature prediction

# Splitting the data into features and targets
X = data[features]
y = data[target_temp]


# Standardize the features using Z-Score Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create sequences and corresponding target values
X_sequences = []
y_sequences = []

for i in range(len(data) - sequence_length - 7):
    X_sequences.append(X_scaled[i:i+sequence_length])
    y_sequences.append(y.iloc[i+sequence_length:i+sequence_length+7].values)

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)



# Calculate the index where the split should occur
split_index = int(len(data) * 0.8)  # 80% train, 20% test

# Splitting into training and testing sets
X_train_scaled, X_test_scaled = X_sequences[:split_index], X_sequences[split_index:]
y_train, y_test = y_sequences[:split_index], y_sequences[split_index:]




# Define the number of splits for time-series cross-validation
n_splits = 5

# Create a TimeSeriesSplit object
tscv = TimeSeriesSplit(n_splits=n_splits)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'units': [32,64,128],
    'num_nodes_per_layer': [[64, 64], [32, 32], [128, 128],[64, 128]],
    'learning_rate': [1e-2,1e-3,1e-4],
    'dropout_rate': [0,0.2,0.4,0.6],  # Dropout rate for input layer (0 for no dropout)
    'num_lstm_layers': [1,2,3]
}

# Build the LSTM model
def build_model(units, learning_rate, dropout_rate, num_lstm_layers,num_nodes_per_layer):
    model = Sequential()
    
    # Input LSTM layer
    model.add(LSTM(units=units, return_sequences=True, input_shape=X_train_scaled.shape[1:]))
    
    # Dropout layer after the input LSTM if dropout_rate > 0
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    
    # Additional LSTM layers
    for _ in range(num_lstm_layers - 1):
        model.add(LSTM(units=units, return_sequences=True))
        # Dropout layer after each additional LSTM layer if dropout_rate_lstm > 0
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
      
    # Additional dropout layer before the Flatten and dense layers
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    
    # Flatten and dense layers (with constant number of nodes)
    model.add(Flatten())
    model.add(Flatten())
    for nodes in num_nodes_per_layer:
        model.add(Dense(units=nodes, activation='relu'))
    
    # Output layer
    model.add(Dense(7))
    
    model.compile(
        loss=MeanSquaredError(),
        optimizer=Adam(learning_rate=learning_rate),
        metrics=[RootMeanSquaredError()]
    )

    return model
# Initialize the TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# Initialize the KerasRegressor with build_model function
keras_regressor = KerasRegressor(build_fn=build_model)

# Create a GridSearchCV object
grid_search = GridSearchCV(
    keras_regressor,
    param_grid,
    cv=tscv,
    n_jobs=-1,
    verbose=2,
)

with tqdm(total=len(param_grid['units']) * len(param_grid['learning_rate']) * len(param_grid['dropout_rate']) * len(param_grid['num_lstm_layers']), desc="Grid Search Progress") as pbar:
    grid_search.fit(X_train_scaled, y_train)  # Use X_train_scaled and y_train
    pbar.update(1)


# Get the best parameters from the grid search
best_params = grid_search.best_params_

print("Best Parameters:", best_params)


# Build the final model using the best parameters
final_best_model = build_model(**best_params)


# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the final best model on the entire dataset
final_best_history = final_best_model.fit(
    X_train_scaled, y_train,  # Use X_train_scaled and y_train
    epochs=50,
    batch_size=32,
    validation_data=(X_test_scaled, y_test)  # Add validation data
)

# Plot average training and validation loss
plt.plot(final_best_history.history['loss'], label='Training Loss')
plt.plot(final_best_history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot predicted vs. actual values
plt.figure(figsize=(10, 6))
for i in range(len(X_test)):
    y_pred = final_best_model.predict(X_test_scaled[i:i+1])[0]
    plt.plot(y_pred, label=f'Fold {i+1} Prediction', linestyle='dashed')
    plt.plot(y_test[i], label=f'Fold {i+1} Actual')
plt.xlabel('Day')
plt.ylabel('Temperature')
plt.title('Predicted vs. Actual Temperature')
plt.legend()
plt.show()

from math import sqrt
from sklearn.metrics import mean_absolute_error

# Evaluate the final model on the test dataset
y_pred = final_best_model.predict(X_test_scaled)
final_mse = mean_squared_error(y_test, y_pred)
final_r2 = r2_score(y_test, y_pred)
final_mae = mean_absolute_error(y_test, y_pred)
final_rmse = sqrt(final_mse)

print(f"Final Mean Squared Error: {final_mse}")
print(f"Final R-squared Score: {final_r2}")
print(f"Final RMSE: {final_rmse}")
print(f"Final MAEW: {final_mae}")

# Train the final best GRU model on the entire dataset
final_best_history = final_best_model.fit(
    X_train_scaled, y_train,  # Use X_train_scaled and y_train
    epochs=50,
    batch_size=32,
    validation_data=(X_test_scaled, y_test)  # Add validation data
)

# Plot average training and validation loss
plt.plot(final_best_history.history['loss'], label='Training Loss')
plt.plot(final_best_history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

"""# Determine the number of data points corresponding to the months
final_best_history = final_best_model.fit(
    X_train_scaled, y_train,  # Use X_train_scaled and y_train
    epochs=5,
    batch_size=64,
    validation_data=(X_test_scaled, y_test)  # Add validation data
)"""

num_months = 12
num_data_points = 30 * num_months  # Assuming an average of 30 data points per month
y_pred = final_best_model.predict(X_test_scaled)
final_mse = mean_squared_error(y_test, y_pred)
final_r2 = r2_score(y_test, y_pred)
final_mae = mean_absolute_error(y_test, y_pred)
final_rmse = sqrt(final_mse)

print(f"Final Mean Squared Error: {final_mse}")
print(f"Final R-squared Score: {final_r2}")
print(f"Final RMSE: {final_rmse}")
print(f"Final MAEW: {final_mae}")

# Calculate the starting index for the subset
start_index = len(y_test) - num_data_points
start_index_1 = len(data) - num_data_points

# Extract the actual and predicted values for the subset
subset_y_test = y_test[start_index:]
subset_y_pred = y_pred[start_index:] 


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

plt.title('Predicted vs. Real Temperature Values (Latest 12 Months)')
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

num_months = 20
num_data_points = 30 * num_months  # Assuming an average of 30 data points per month
y_pred = final_best_model.predict(X_test_scaled)
final_mse = mean_squared_error(y_test, y_pred)
final_r2 = r2_score(y_test, y_pred)
final_mae = mean_absolute_error(y_test, y_pred)
final_rmse = sqrt(final_mse)

print(f"Final Mean Squared Error: {final_mse}")
print(f"Final R-squared Score: {final_r2}")
print(f"Final RMSE: {final_rmse}")
print(f"Final MAEW: {final_mae}")

# Calculate the starting index for the subset
start_index = len(y_test) - num_data_points
start_index_1 = len(data) - num_data_points

# Extract the actual and predicted values for the subset
subset_y_test = y_test[start_index:]
subset_y_pred = y_pred[start_index:] 


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

plt.title('Predicted vs. Real Temperature Values (Latest 12 Months)')
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

