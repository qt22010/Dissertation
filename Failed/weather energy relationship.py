# -*- coding: utf-8 -*-


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load weather and energy datasets
weather_data = pd.read_csv("weather_dataset.csv")
energy_data = pd.read_csv("energy_dataset.csv")

# Merge datasets on the 'Date' column
merged_data = pd.merge(energy_data, weather_data, on="Date")

# Calculate daily energy consumption across all households
daily_energy = merged_data.groupby("Date")["KWH"].sum()

# Exploratory Data Analysis (EDA)
# Visualize energy consumption over time
plt.figure(figsize=(12, 6))
plt.plot(daily_energy.index, daily_energy.values)
plt.xlabel("Date")
plt.ylabel("Energy Consumption (KWH)")
plt.title("Daily Energy Consumption")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Correlation Analysis
correlation_matrix = merged_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Visualize relationship between weather features and energy consumption
plt.figure(figsize=(14, 10))
selected_features = ["mean_temperature", "humidity", "wind_speed", "clouds"]
for i, feature in enumerate(selected_features, 1):
    plt.subplot(2, 2, i)
    plt.scatter(merged_data[feature], merged_data["KWH"], alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel("Energy Consumption (KWH)")
    plt.title(f"Relationship between {feature.capitalize()} and Energy Consumption")
plt.tight_layout()
plt.show()

# Box plots to show energy consumption across different levels of a weather feature DOES THIS MAKE SENSE EVEN?
plt.figure(figsize=(12, 6))
selected_boxplot_features = ["humidity", "clouds"]
for i, feature in enumerate(selected_boxplot_features, 1):
    plt.subplot(1, 2, i)
    sns.boxplot(x=merged_data[feature], y=merged_data["KWH"])
    plt.xlabel(feature)
    plt.ylabel("Energy Consumption (KWH)")
    plt.title(f"Energy Consumption Variation with {feature.capitalize()}")
plt.tight_layout()
plt.show()


# Time-Series Decomposition
# Apply time-series decomposition (e.g., seasonal decomposition) to better understand seasonality, trends, and residuals.

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load weather and energy datasets
weather_data = pd.read_csv("weather_dataset.csv")
energy_data = pd.read_csv("energy_dataset.csv")

# Merge datasets on the 'Date' column
merged_data = pd.merge(energy_data, weather_data, on="Date")

# Calculate daily energy consumption across all households
daily_energy_sum = merged_data.groupby("Date")["KWH"].sum()
daily_energy_mean = merged_data.groupby("Date")["KWH"].mean()

# Perform time series decomposition for sum
decomposition_sum = seasonal_decompose(daily_energy_sum, model='additive')

# Perform time series decomposition for mean
decomposition_mean = seasonal_decompose(daily_energy_mean, model='additive')

# Plot the decomposition components for sum
plt.figure(figsize=(12, 16))

# Observed
plt.subplot(6, 1, 1)
plt.plot(decomposition_sum.observed)
plt.ylabel("Observed (Sum)")
plt.title("Time Series Decomposition (Sum)")

# Trend
plt.subplot(6, 1, 2)
plt.plot(decomposition_sum.trend)
plt.ylabel("Trend")

# Seasonal
plt.subplot(6, 1, 3)
plt.plot(decomposition_sum.seasonal)
plt.ylabel("Seasonal")

# Residual
plt.subplot(6, 1, 4)
plt.plot(decomposition_sum.resid)
plt.ylabel("Residual")

plt.tight_layout()
plt.show()

# Plot the decomposition components for mean
plt.figure(figsize=(12, 16))

# Observed
plt.subplot(6, 1, 1)
plt.plot(decomposition_mean.observed)
plt.ylabel("Observed (Mean)")
plt.title("Time Series Decomposition (Mean)")

# Trend
plt.subplot(6, 1, 2)
plt.plot(decomposition_mean.trend)
plt.ylabel("Trend")

# Seasonal
plt.subplot(6, 1, 3)
plt.plot(decomposition_mean.seasonal)
plt.ylabel("Seasonal")

# Residual
plt.subplot(6, 1, 4)
plt.plot(decomposition_mean.resid)
plt.ylabel("Residual")

plt.tight_layout()
plt.show()

# Weekday vs. Weekend Analysis
# Compare energy consumption patterns on weekdays and weekends to see if there are differences based on weather conditions.

import pandas as pd
import matplotlib.pyplot as plt

# Load weather and energy datasets
weather_data = pd.read_csv("weather_dataset.csv")
energy_data = pd.read_csv("energy_dataset.csv")

# Merge datasets on the 'Date' column
merged_data = pd.merge(energy_data, weather_data, on="Date")

# Calculate daily energy consumption across all households
daily_energy = merged_data.groupby("Date")["KWH"].sum()

# Add a 'DayOfWeek' column to indicate weekday (0-4) or weekend (5-6)
daily_energy['DayOfWeek'] = daily_energy.index.dayofweek

# Group by 'DayOfWeek' and calculate sum and mean energy consumption
sum_energy_by_day = daily_energy.groupby("DayOfWeek")["KWH"].sum()
mean_energy_by_day = daily_energy.groupby("DayOfWeek")["KWH"].mean()

# Plot sum energy consumption on weekdays and weekends
plt.figure(figsize=(12, 6))
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
plt.subplot(1, 2, 1)
plt.bar(days, sum_energy_by_day, color=['blue', 'blue', 'blue', 'blue', 'blue', 'green', 'green'])
plt.xlabel("Day of Week")
plt.ylabel("Total Energy Consumption (KWH)")
plt.title("Total Energy Consumption: Weekdays vs. Weekends")
plt.xticks(rotation=45)

# Plot mean energy consumption on weekdays and weekends
plt.subplot(1, 2, 2)
plt.bar(days, mean_energy_by_day, color=['blue', 'blue', 'blue', 'blue', 'blue', 'green', 'green'])
plt.xlabel("Day of Week")
plt.ylabel("Average Energy Consumption (KWH)")
plt.title("Average Energy Consumption: Weekdays vs. Weekends")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# Interaction Effects/multivarate analysis
# Explore interactions between weather features (e.g., humidity and temperature) to see if combined effects impact energy consumption differently.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load weather and energy datasets
weather_data = pd.read_csv("weather_dataset.csv")
energy_data = pd.read_csv("energy_dataset.csv")

# Merge datasets on the 'Date' column
merged_data = pd.merge(energy_data, weather_data, on="Date")

# Select relevant weather features
selected_features = ["mean_temperature", "humidity", "wind_speed", "clouds"]

# Create a scatterplot matrix to visualize interaction effects
sns.pairplot(merged_data, vars=selected_features)
plt.suptitle("Interaction Effects between Weather Features", y=1.02)
plt.show()



# Cluster Analysis
# Use clustering techniques to group similar weather-energy consumption patterns.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load weather and energy datasets
weather_data = pd.read_csv("weather_dataset.csv")
energy_data = pd.read_csv("energy_dataset.csv")

# Merge datasets on the 'Date' column
merged_data = pd.merge(energy_data, weather_data, on="Date")

# Select relevant weather features for clustering
selected_features = ["mean_temperature", "humidity", "wind_speed", "clouds"]

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(merged_data[selected_features])

# Determine optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Plot the Elbow curve
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal Number of Clusters")
plt.show()

