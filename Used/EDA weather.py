
import requests
import pandas as pd
from datetime import datetime
import time
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split



weather_df = pd.read_csv(r'C:\Users\white\Desktop\Dissertation\data\weather.csv')

# Display the first few rows of the DataFrame
print(weather_df.head())


print(weather_df.info())

# Parse the 'dt_iso' column as datetime
weather_df['date'] = pd.to_datetime(weather_df['dt_iso'].str[:19], format='%Y-%m-%d %H:%M:%S')

weather_df=weather_df.drop(weather_df[weather_df['date'] == '2023-07-23 11:00:00'].index)
# Set 'date' column as index
weather_df.set_index('date', inplace=True)


# Define the desired columns
desired_columns = [
    'temp',
    'dew_point',
    'feels_like',
    'temp_min',
    'temp_max',
    'pressure',
    'humidity',
    'wind_speed',
    'wind_deg',
    'rain_1h',
    'snow_1h',
    'clouds_all'
]

# Keep only the desired columns
weather_df = weather_df[desired_columns]

# Set up aggregation functions for precipitation variables
agg_precipitation = {
    'rain_1h': 'sum',
    'snow_1h': 'sum'
}

# Set up aggregation functions for temperature-related variables
agg_temperature = {
    'temp': 'mean',  # or 'median'
    'temp_min': 'min',
    'temp_max': 'max',
    'feels_like': 'mean'  # or 'median'
}

# Set up aggregation functions for other variables
agg_pressure_humidity_wind = {
    'pressure': 'mean',  # or 'median'
    'humidity': 'mean',
    'wind_speed': 'mean',  # or 'median'
    'wind_deg': lambda x: x.mode().iloc[0] if not x.empty else None,  # mode for wind direction
    'clouds_all': 'mean',  # or 'median'
    'dew_point': 'mean'  # or 'median'
}

# Combine all the aggregation functions
agg_funcs = {**agg_precipitation, **agg_temperature, **agg_pressure_humidity_wind}

# Resample the data to daily frequency and apply aggregation functions
weather_data = weather_df.resample('D').agg(agg_funcs)


# Convert the index to have only the date (without the time component)
weather_data.index = weather_data.index.date
# Print the aggregated daily weather DataFrame
print(weather_data.head())

print(weather_data)

print(weather_data.info())

# Display summary statistics
print(weather_data.describe())

# Check for missing values
print(weather_data.isnull().sum())


weather_data['date'] = weather_data.index



#Exploratory Data Analysis (EDA) - Visualizations and Correlations
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np


# Correlation Heatmap
corr_matrix = weather_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='plasma', center=0)
plt.title("Correlation Heatmap for Weather features")
plt.show()

# Select the desired columns
selected_columns = ['temp','dew_point', 'temp_min', 'temp_max', 'humidity', 'clouds_all', 'snow_1h']

# Create the new DataFrame temp_weather
temp_weather = weather_data[selected_columns]

# Now you have a DataFrame called temp_weather with the selected features
print(temp_weather)

temp_weather=temp_weather.reset_index(inplace=False)
temp_weather = temp_weather.rename(columns={'index': 'date'})

temp_weather['date'] = pd.to_datetime(temp_weather['date'])
# Extract day of the week, month, and year from the 'date' column
temp_weather['day_of_week'] = temp_weather['date'].dt.dayofweek
temp_weather['month'] = temp_weather['date'].dt.month
temp_weather['year'] = temp_weather['date'].dt.year

# Create a season feature based on the month
temp_weather['season'] = (temp_weather['month'] % 12 + 3) // 3  # Calculate the season based on month

# Map season numbers to season names
season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
temp_weather['season_name'] = temp_weather['season'].map(season_map)

# Print the updated DataFrame
print(temp_weather)

# Select the desired columns
selected_columns = ['dew_point', 'wind_speed', 'pressure', 'humidity', 'clouds_all', 'rain_1h']

# Create the new DataFrame temp_weather
rain_weather = weather_data[selected_columns]

# Now you have a DataFrame called temp_weather with the selected features
print(rain_weather)

rain_weather=rain_weather.reset_index(inplace=False)
rain_weather=rain_weather.rename(columns={'index': 'date'})
rain_weather['date'] = pd.to_datetime(rain_weather['date'])
# Extract day of the week, month, and year from the 'date' column
rain_weather['day_of_week'] = rain_weather['date'].dt.dayofweek
rain_weather['month'] = rain_weather['date'].dt.month
rain_weather['year'] = rain_weather['date'].dt.year

# Create a season feature based on the month
rain_weather['season'] = (rain_weather['month'] % 12 + 3) // 3  # Calculate the season based on month

# Map season numbers to season names
rain_weather['season_name'] = rain_weather['season'].map(season_map)

print(rain_weather)
temp_weather.to_csv(r'C:\Users\white\Desktop\Dissertation\data\temp_weather.csv')

rain_weather.to_csv(r'C:\Users\white\Desktop\Dissertation\data\rain_weather.csv')


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Calculate the average temperature for each month and year
monthly_avg_temp_yearly = temp_weather.groupby(['year', 'month'])['temp'].mean().reset_index()
monthly_avg_temp = temp_weather.groupby('month')['temp'].mean().reset_index()

# Map season numbers to season names based on month ranges
season_map = {(12, 1, 2): 'Winter', (3, 4, 5): 'Spring', (6, 7, 8): 'Summer', (9, 10, 11): 'Fall'}
monthly_avg_temp_yearly['season_name'] = monthly_avg_temp_yearly['month'].apply(lambda x: next(season for months, season in season_map.items() if x in months))

# Plotting the average temperature per month across all years
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_avg_temp, x='month', y='temp')
plt.title('Average Monthly Temperature Patterns Across All Years')
plt.xlabel('Month')
plt.ylabel('Average Temperature')
plt.xticks(np.arange(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)

plt.show()
daily_avg_temp = temp_weather.groupby('day_of_week')['temp'].mean().reset_index()
# Plotting the average temperature per day across all years
# Plotting the average temperature per day across all years
plt.figure(figsize=(12, 6))

# Increase font size for the plot
sns.set_context("paper", rc={"font.size": 14})  # Adjust the font size as needed

sns.lineplot(data=daily_avg_temp, x='day_of_week', y='temp')
plt.title('Average Daily Temperature Patterns Across All Years', fontsize=16)  # Increase title font size
plt.xlabel('Day of the Week', fontsize=14)  # Increase x-axis label font size
plt.ylabel('Average Temperature', fontsize=16)  # Increase y-axis label font size
plt.xticks(np.arange(0, 7),['Mon', 'Tues', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], fontsize=12)  # Increase tick label font size
plt.grid(True)

plt.show()

# Plotting the average temperature for each season across the years
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_avg_temp_yearly, x='year', y='temp', hue='season_name')
plt.title('Average Temperature for Each Season Across the Years')
plt.xlabel('Year')
plt.ylabel('Average Temperature')
plt.grid(True)
plt.legend(title='Season')
plt.show()



# Calculate the average temperature per year
yearly_avg_temp = temp_weather.groupby('year')['temp'].mean().reset_index()

plt.figure(figsize=(12, 6))

# Plot a single straight line for yearly average temperatures
plt.plot(yearly_avg_temp['year'], yearly_avg_temp['temp'], color='red', linestyle='--', marker='o', markersize=8, label='Yearly Avg Temp')

plt.title('Average Yearly Temperature')
plt.xlabel('Year')
plt.ylabel('Average Temperature')
plt.grid(True)
plt.legend()
plt.show()

# Calculate the average temperature for each month
monthly_avg_temp = temp_weather.groupby('month')['temp'].mean().reset_index()

# Map season numbers to season names based on month ranges
season_map = {(12, 1, 2): 'Winter', (3, 4, 5): 'Spring', (6, 7, 8): 'Summer', (9, 10, 11): 'Fall'}
monthly_avg_temp['season_name'] = monthly_avg_temp['month'].apply(lambda x: next(season for months, season in season_map.items() if x in months))

print(monthly_avg_temp)
# Plotting the average temperature patterns with color-coded by season using a bar chart
plt.figure(figsize=(12, 6))
sns.barplot(data=monthly_avg_temp, x='month', y='temp', hue='season_name')
plt.title('Average Monthly Temperature Patterns Color-coded by Season')
plt.xlabel('Month')
plt.ylabel('Average Temperature')
plt.xticks(np.arange(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)
plt.legend(title='Season')
plt.show()


# Distribution Plots
plt.figure(figsize=(12, 8))
sns.histplot(data=weather_data, x='temp', kde=True, color='blue', bins=20)
plt.title("Distribution of Temperature")
plt.xlabel("Temperature")
plt.ylabel("Frequency")
plt.show()


# Set 'date' as the index
weather_data.set_index('date', inplace=True)

from statsmodels.tsa.seasonal import seasonal_decompose #HERE
temp_weather.index.freq = 'D'
# Decompose temperature data
result = seasonal_decompose(temp_weather['temp'], model='additive', period=90)

# Define the date window
start_date = pd.to_datetime('2010-01-01')
end_date = pd.to_datetime('2012-12-31')
# Filter data within the date window
filtered_weather_data = weather_data[start_date:end_date]
# Decompose temperature data using seasonal_decompose with period parameter
result = seasonal_decompose(filtered_weather_data['temp'], model='additive', period=90)

# Plot decomposition components
plt.figure(figsize=(12, 8))
result.plot()
plt.show()

# Select the desired columns
selected_columns = ['temp','dew_point', 'temp_min', 'temp_max', 'humidity', 'clouds_all', 'snow_1h']
sns.pairplot(temp_weather[selected_columns])
plt.suptitle('Pairwise Relationships between Features', y=1.02)
plt.show()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Plot ACF and PACF for temperature data
plt.figure(figsize=(12, 6))
plot_acf(filtered_weather_data['temp'], lags=200, title='ACF - Temperature')
plt.show()

plt.figure(figsize=(12, 6))
plot_pacf(filtered_weather_data['temp'], lags=200, title='PACF - Temperature')
plt.show()






#rain

# Calculate the average rainfall for each season, year, and month
monthly_avg_rain_yearly_monthly = rain_weather.groupby(['year', 'season_name', 'month'])['rain_1h'].mean().reset_index()

# Plotting the average rainfall for each season across the years by month
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_avg_rain_yearly_monthly, x='month', y='rain_1h')
plt.title('Average Rainfall for Each Month Across the Years')
plt.xlabel('Month')
plt.ylabel('Average Rainfall (mm)')
plt.grid(True)
plt.xticks(np.arange(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()


# Calculate the average rainfall per year
yearly_avg_rain = rain_weather.groupby('year')['rain_1h'].mean().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(yearly_avg_rain['year'], yearly_avg_rain['rain_1h'], marker='o')
plt.title('Average Rainfall Over the Years')
plt.xlabel('Year')
plt.ylabel('Average Rainfall (mm)')
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Create a boxplot for each month across all years
plt.figure(figsize=(12, 6))

# Create a boxplot showing only the IQR
sns.boxplot(data=temp_weather, x='month', y='temp', showfliers=False)

plt.title('Temperature Distribution by Month (IQR Only)')
plt.xlabel('Month')
plt.ylabel('Temperature')
plt.xticks(np.arange(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)
plt.show()



# Calculate the average rainfall for each month and year
monthly_avg_rain_yearly = rain_weather.groupby(['year', 'month'])['rain_1h'].mean().reset_index()
monthly_avg_rain = rain_weather.groupby('month')['rain_1h'].mean().reset_index()

# Map season numbers to season names based on month ranges
season_map = {(12, 1, 2): 'Winter', (3, 4, 5): 'Spring', (6, 7, 8): 'Summer', (9, 10, 11): 'Fall'}
monthly_avg_rain_yearly['season_name'] = monthly_avg_rain_yearly['month'].apply(lambda x: next(season for months, season in season_map.items() if x in months))

# Plotting the average rainfall per month across all years
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_avg_rain, x='month', y='rain_1h')
plt.title('Average Monthly Rainfall Patterns Across All Years')
plt.xlabel('Month')
plt.ylabel('Average Rainfall')
plt.xticks(np.arange(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)

plt.show()

daily_avg_rain = rain_weather.groupby('day_of_week')['rain_1h'].mean().reset_index()
# Plotting the average rainfall per day across all years
plt.figure(figsize=(12, 6))

# Increase font size for the plot
sns.set_context("paper", rc={"font.size": 14})  # Adjust the font size as needed

sns.lineplot(data=daily_avg_rain, x='day_of_week', y='rain_1h')
plt.title('Average Daily Rainfall Patterns Across All Years', fontsize=16)  # Increase title font size
plt.xlabel('Day of the Week', fontsize=14)  # Increase x-axis label font size
plt.ylabel('Average Rainfall', fontsize=16)  # Increase y-axis label font size
plt.xticks(np.arange(0, 7),['Mon', 'Tues', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], fontsize=12)  # Increase tick label font size
plt.grid(True)

plt.show()

# Plotting the average rainfall for each season across the years
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_avg_rain_yearly, x='year', y='rain_1h', hue='season_name')
plt.title('Average Rainfall for Each Season Across the Years')
plt.xlabel('Year')
plt.ylabel('Average Rainfall')
plt.grid(True)
plt.legend(title='Season')
plt.show()
