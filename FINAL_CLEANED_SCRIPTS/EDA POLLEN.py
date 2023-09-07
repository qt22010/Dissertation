# -*- coding: utf-8 -*-
"""
@author: white
"""

import pandas as pd
import os

excel_directory = r'C:\Users\white\Desktop\Dissertation\data\pollen'
merged_csv_path = r'C:\Users\white\Desktop\Dissertation\data\pollen\pollen_merged_file.csv'

excel_files = [file for file in os.listdir(excel_directory) if file.endswith('.xlsx')]

merged_data = pd.DataFrame()

for excel_file in excel_files:
    file_path = os.path.join(excel_directory, excel_file)
    data = pd.read_excel(file_path)
    merged_data = pd.concat([merged_data, data], ignore_index=True)

merged_data.to_csv(merged_csv_path, index=False)


# Load the pollen count dataset and the air quality dataset
pollen_data = pd.read_csv(r'C:\Users\white\Desktop\Dissertation\data\pollen\pollen_merged_file.csv')
air_quality_data = pd.read_csv(r'C:\Users\white\Desktop\Dissertation\data\pollen\london-air-quality.csv')

print(pollen_data)

# Convert date columns to datetime format
pollen_data['date'] = pd.to_datetime(pollen_data['date'], format='%d/%m/%Y')
air_quality_data['date'] = pd.to_datetime(air_quality_data['date'], format='%d/%m/%Y')

# Filter air quality data based on dates in the pollen count dataset
filtered_air_quality = air_quality_data[air_quality_data['date'].isin(pollen_data['date'])]

# Merge filtered air quality data with pollen count data based on date
merged_data1 = pd.merge(pollen_data, filtered_air_quality, on='date', how='left')

print(merged_data1)

# Convert the 'date' column to datetime format
merged_data1['date'] = pd.to_datetime(merged_data1['date'])

# Create a boolean mask to filter rows before 2014
mask = merged_data1['date'].dt.year >= 2014

# Apply the mask to the DataFrame to keep rows from 2014 onwards
filtered_merged_data1 = merged_data1[mask]

# Now you have the filtered DataFrame without rows before 2014
print(filtered_merged_data1)

# Save the merged data to a new CSV file
filtered_merged_data1.to_csv(r'C:\Users\white\Desktop\Dissertation\data\pollen\pollen_air.csv', index=False)


# Load the pollen count dataset and the weather dataset
pollen_data = pd.read_csv(r'C:\Users\white\Desktop\Dissertation\data\pollen\pollen_air.csv')
weather_data = pd.read_csv(r'C:\Users\white\Desktop\Dissertation\data\weather_data_cleaned1.csv')

# Define a function to parse dates with fallback for different formats
def parse_date(date_str):
    return pd.to_datetime(date_str, format='%Y-%m-%d')  # Try year-month-day format

# Convert date columns to datetime format using the custom parsing function
pollen_data['date'] = pollen_data['date'].apply(parse_date)
weather_data['date'] = pd.to_datetime(weather_data['date'], format='%d/%m/%Y')  # Assuming weather data is in day/month/year format

# Filter weather data based on dates in the pollen count dataset
filtered_weather = weather_data[weather_data['date'].isin(pollen_data['date'])]

# Merge filtered weather data with pollen count data based on date
final_pollen = pd.merge(pollen_data, filtered_weather[['date', 'temp', 'humidity', 'wind_speed','clouds_all','dew_point']], on='date', how='left')

# Save the merged data to a new CSV file
final_pollen.to_csv(r'C:\Users\white\Desktop\Dissertation\data\pollen\pollen_weather.csv', index=False)



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
final_pollen=pd.read_csv(r'C:\Users\white\Desktop\Dissertation\data\pollen\pollen_weather.csv')
merged_dataset=final_pollen
# Basic statistics and structure of the dataset
print(merged_dataset.info())
print(merged_dataset.describe())

missing_values = merged_dataset.isna().sum()

# Print the count of missing values for each column
print(missing_values)


import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the correlation matrix
corr_matrix = merged_dataset.corr()
print(corr_matrix)
# Create a correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='plasma', center=0)
plt.title("Correlation Heatmap for Pollen Features")
plt.show()


pollen_final=pd.read_csv(r'C:\Users\white\Desktop\Dissertation\data\pollen\pollen_weather.csv')

# Convert 'date' column to datetime format (assuming format is 'D/M/Y')
pollen_final['date'] = pd.to_datetime(pollen_final['date'], format='%d/%m/%Y')

# Extract temporal features
pollen_final['day_of_week'] = pollen_final['date'].dt.dayofweek
pollen_final['month'] = pollen_final['date'].dt.month
pollen_final['year'] = pollen_final['date'].dt.year
pollen_final['season'] = (pollen_final['month'] % 12 + 3) // 3

# Map season numbers to season names
season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
pollen_final['season_name'] = pollen_final['season'].map(season_map)

# Print the updated DataFrame
print(pollen_final)



# Pairwise correlation heatmap
correlation_matrix = merged_dataset.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# Convert 'date' column to datetime format
merged_dataset['date'] = pd.to_datetime(merged_dataset['date'], dayfirst=True)

# Extract month from date
merged_dataset['month'] = merged_dataset['date'].dt.month

# Calculate average pollen count per month
average_pollen_by_month = merged_dataset.groupby('month')['Poac'].mean()


# Distribution of other features
plt.figure(figsize=(12, 6))
sns.boxplot(data=merged_dataset[['temp', 'humidity', 'wind_speed', ' o3', ' pm25', ' pm10']])
plt.title('Distribution of Features')
plt.xticks(rotation=45)
plt.show()

# Handling Missing Data
missing_values = merged_dataset.isnull().sum()
print("Missing Values:")
print(missing_values)


import matplotlib.pyplot as plt
import seaborn as sns

# Extract year and month from date
merged_dataset['year'] = merged_dataset['date'].dt.year
merged_dataset['month'] = merged_dataset['date'].dt.month

# Create a seasonal plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=merged_dataset, x='month', y='Poac', hue='year', marker='o')
plt.title('Seasonal Plot of Pollen Count')
plt.xlabel('Month')
plt.ylabel('Pollen Count')
plt.xticks(range(1, 13))
plt.tight_layout()
plt.show()

# Pivot the data to create a heatmap
heatmap_data = merged_dataset.pivot_table(index='month', columns='year', values='Poac', aggfunc='mean')

# Create a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt=".1f")
plt.title('Pollen Count Heatmap by Month and Year')
plt.xlabel('Year')
plt.ylabel('Month')
plt.tight_layout()
plt.show()

#########################

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(pollen_final['Poac'], bins=20, edgecolor='black')
plt.title('Distribution of Pollen Counts')
plt.xlabel('Pollen Count')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(pollen_final['date'], pollen_final['Poac'])
plt.title('Pollen Count Variation Over Time (Spring and Summer)')
plt.xlabel('Date')
plt.ylabel('Pollen Count')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(data=pollen_final, x='season_name', y='Poac')
plt.title('Pollen Count Distribution by Season')
plt.xlabel('Season')
plt.ylabel('Pollen Count')
plt.show()




import numpy as np

# Calculate the average pollen count for each month and year
monthly_avg_pollen_yearly = pollen_final.groupby(['year', 'month'])['Poac'].mean().reset_index()
monthly_avg_pollen = pollen_final.groupby('month')['Poac'].mean().reset_index()

# Map season numbers to season names based on month ranges
season_map = {(12, 1, 2): 'Winter', (3, 4, 5): 'Spring', (6, 7, 8): 'Summer', (9, 10, 11): 'Fall'}
monthly_avg_pollen_yearly['season_name'] = monthly_avg_pollen_yearly['month'].apply(lambda x: next(season for months, season in season_map.items() if x in months))

# Plotting the average pollen count per month across all years
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_avg_pollen, x='month', y='Poac')
plt.title('Average Monthly Pollen Count Patterns Across All Years')
plt.xlabel('Month')
plt.ylabel('Average Pollen Count')
plt.xticks(np.arange(3, 10), ['Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'])
plt.grid(True)

plt.show()

daily_avg_pollen = pollen_final.groupby('day_of_week')['Poac'].mean().reset_index()
# Plotting the average pollen count per day across all years
plt.figure(figsize=(12, 6))

# Increase font size for the plot
sns.set_context("paper", rc={"font.size": 14})  # Adjust the font size as needed

sns.lineplot(data=daily_avg_pollen, x='day_of_week', y='Poac')
plt.title('Average Daily Pollen Count Patterns Across All Years', fontsize=16)  # Increase title font size
plt.xlabel('Day of the Week', fontsize=14)  # Increase x-axis label font size
plt.ylabel('Average Pollen Count', fontsize=16)  # Increase y-axis label font size
plt.xticks(np.arange(0, 7),['Mon', 'Tues', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], fontsize=12)  # Increase tick label font size
plt.grid(True)

plt.show()

# Plotting the average pollen count for each season across the years
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_avg_pollen_yearly, x='year', y='Poac', hue='season_name')
plt.title('Average Pollen Count for Each Season Across the Years')
plt.xlabel('Year')
plt.ylabel('Average Pollen Count')
plt.grid(True)
plt.legend(title='Season')
plt.show()



# Calculate the average pollen count for each year
yearly_avg_pollen = pollen_final.groupby('year')['Poac'].mean().reset_index()

# Plotting the average pollen count per year
plt.figure(figsize=(12, 6))
sns.lineplot(data=yearly_avg_pollen, x='year', y='Poac')
plt.title('Average Pollen Count per Year')
plt.xlabel('Year')
plt.ylabel('Average Pollen Count')
plt.grid(True)

plt.show()

plt.figure(figsize=(12, 6))
plt.plot(yearly_avg_pollen['year'], yearly_avg_pollen['Poac'], marker='o')
plt.title('Average Pollen Count Over the Years')
plt.xlabel('Year')
plt.ylabel('Average Pollen Count')
plt.grid(True)
plt.show()

# Plot a single straight line for yearly average pollen counts
plt.plot(yearly_avg_pollen['year'], yearly_avg_pollen['Poac'], color='green', linestyle='--', marker='o', markersize=4, label='Yearly Avg Pollen Count')

plt.title('Average Yearly Pollen Count')
plt.xlabel('Year')
plt.ylabel('Average Pollen Count')
plt.grid(True)
plt.legend()
plt.show()