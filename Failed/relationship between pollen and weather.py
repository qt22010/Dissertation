# -*- coding: utf-8 -*-


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load weather, air quality, and pollen count datasets
weather_data = pd.read_csv("weather_dataset.csv")
air_quality_data = pd.read_csv("air_quality_dataset.csv")
pollen_count_data = pd.read_csv("pollen_count_dataset.csv")

# Merge datasets on the 'Date' column
merged_data = pd.merge(weather_data, air_quality_data, on="Date")
merged_data = pd.merge(merged_data, pollen_count_data, on="Date")

# Select relevant features
selected_features = ["mean_temperature", "humidity", "wind_speed", "pm25", "pm10", "no2", "o3", "pollen_count"]

# Create a pairplot to visualize relationships
sns.pairplot(merged_data, vars=selected_features, diag_kind="kde")
plt.suptitle("Relationships between Weather, Air Quality, and Pollen Count", y=1.02)
plt.show()

# Calculate correlation matrix
correlation_matrix = merged_data[selected_features].corr()

# Create a heatmap to visualize correlations
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix: Weather, Air Quality, and Pollen Count")
plt.show()



"""Correlation with Lagged Weather and Air Quality:
Calculate correlations between the current day's pollen count and lagged (previous day's) weather and air quality features. 
    This can help you understand if past conditions have an impact on pollen count.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load merged dataset containing weather, air quality, and pollen count data
merged_data = pd.read_csv("pollen_count_dataset.csv")

# Select relevant features for analysis
selected_features = ["mean_temperature", "humidity", "wind_speed", "pm25", "pm10", "no2", "o3", "pollen_count"]

# Calculate lagged features by shifting data one day back
lagged_data = merged_data[selected_features].shift(1)
lagged_data.columns = [f"{col}_lagged" for col in selected_features]

# Concatenate original and lagged data
combined_data = pd.concat([merged_data[selected_features], lagged_data], axis=1)

# Calculate correlations between current day's pollen count and lagged features
correlation_matrix = combined_data.corr()
correlation_with_pollen = correlation_matrix["pollen_count"].drop("pollen_count")

# Plot correlations
plt.figure(figsize=(10, 6))
sns.barplot(x=correlation_with_pollen.index, y=correlation_with_pollen.values)
plt.xticks(rotation=45)
plt.xlabel("Lagged Features")
plt.ylabel("Correlation with Pollen Count")
plt.title("Correlation between Lagged Features and Pollen Count")
plt.tight_layout()
plt.show()


"""Subplots of Relationships:
Create a grid of subplots to visualize the relationships between each weather and air quality feature with pollen count individually.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load merged dataset containing weather, air quality, and pollen count data
merged_data = pd.read_csv("pollen_count_dataset.csv")

# Select relevant features for analysis
selected_features = ["mean_temperature", "humidity", "wind_speed", "pm25", "pm10", "no2", "o3"]

# Create a grid of subplots
plt.figure(figsize=(12, 10))
for i, feature in enumerate(selected_features, 1):
    plt.subplot(3, 3, i)
    sns.scatterplot(x=feature, y="pollen_count", data=merged_data)
    plt.title(f"Relationship between {feature.capitalize()} and Pollen Count")
    plt.xlabel(feature.capitalize())
    plt.ylabel("Pollen Count")
plt.tight_layout()
plt.show()

"""Time series decomposition is applied to each selected feature within this filtered period, and the decomposition components (observed, trend, 
seasonal, residual) are plotted for each feature. This analysis will help you identify seasonal patterns, trends, and residuals within the pollen 
 count and weather features during the specified months."""
 
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load merged dataset containing weather, air quality, and pollen count data
merged_data = pd.read_csv("pollen_count_dataset.csv")

# Convert the 'Date' column to datetime format
merged_data['Date'] = pd.to_datetime(merged_data['Date'])

# Select relevant features for analysis
selected_features = ["mean_temperature", "humidity", "wind_speed", "pm25", "pm10", "no2", "o3", "pollen_count"]

# Filter data for the months of late April to early September
start_date = pd.Timestamp(year=merged_data['Date'].min().year, month=4, day=20)
end_date = pd.Timestamp(year=merged_data['Date'].max().year, month=9, day=10)
filtered_data = merged_data[(merged_data['Date'] >= start_date) & (merged_data['Date'] <= end_date)]

# Apply time series decomposition for each selected feature
plt.figure(figsize=(12, 8))
for feature in selected_features:
    decomposition = seasonal_decompose(filtered_data[feature], model='additive')
    
    plt.subplot(len(selected_features), 1, selected_features.index(feature) + 1)
    plt.plot(decomposition.observed, label='Observed')
    plt.plot(decomposition.trend, label='Trend')
    plt.plot(decomposition.seasonal, label='Seasonal')
    plt.plot(decomposition.resid, label='Residual')
    plt.title(f"Time Series Decomposition of {feature.capitalize()}")
    plt.ylabel("Value")
    plt.xlabel("Date")
    plt.legend()
    
plt.tight_layout()
plt.show()


"""Long-Term Trends:
Extend your long-term trend analysis to cover the 10-year period from late April to early September. This will help you identify any 
significant changes or shifts in pollen count, weather, and air quality patterns over the years.
"""
import pandas as pd
import matplotlib.pyplot as plt

# Load merged dataset containing weather, air quality, and pollen count data
merged_data = pd.read_csv("pollen_count_dataset.csv")

# Convert the 'Date' column to datetime format
merged_data['Date'] = pd.to_datetime(merged_data['Date'])

# Filter data for the months of late April to early September
start_date = pd.Timestamp(year=merged_data['Date'].min().year, month=4, day=20)
end_date = pd.Timestamp(year=merged_data['Date'].max().year, month=9, day=10)
filtered_data = merged_data[(merged_data['Date'] >= start_date) & (merged_data['Date'] <= end_date)]

# List of pollen and weather feature columns
pollen_features = ["feature1", "feature2", "feature3", "feature4"]  # Replace with actual pollen features
weather_features = ["mean_temperature", "humidity", "wind_speed", "rain", "dew_point", "pressure"]  # Replace with actual weather features

# Calculate average pollen and weather features by year
grouped_data = filtered_data.groupby(filtered_data['Date'].dt.year)[pollen_features + weather_features].mean()

# Plot long-term trends for pollen and weather features
plt.figure(figsize=(12, 8))

# Plot pollen features
for feature in pollen_features:
    plt.plot(grouped_data.index, grouped_data[feature], label=feature.replace("_", " ").capitalize(), linestyle='--')

# Plot weather features
for feature in weather_features:
    plt.plot(grouped_data.index, grouped_data[feature], label=feature.replace("_", " ").capitalize())

plt.xlabel("Year")
plt.ylabel("Average Value")
plt.title("Long-Term Trends in Pollen and Weather Features")
plt.legend()
plt.tight_layout()
plt.show()


"""
Cluster Analysis and Segment Patterns:
With a longer time span, consider applying cluster analysis to group years with similar pollen count, weather, and air quality patterns. 
This can help you identify distinct periods or trends within the 10-year dataset.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load merged dataset containing weather, air quality, and pollen count data
merged_data = pd.read_csv("pollen_count_dataset.csv")

# Convert the 'Date' column to datetime format
merged_data['Date'] = pd.to_datetime(merged_data['Date'])

# List of features for clustering
features_for_clustering = ["mean_temperature", "humidity", "wind_speed", "rain", "dew_point", "pressure", "pollen_count"]  # Include relevant features

# Prepare data for clustering
data_for_clustering = merged_data[features_for_clustering].values

# Determine the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):  # Try different numbers of clusters
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(data_for_clustering)
    wcss.append(kmeans.inertia_)

# Plot the elbow method to find the optimal number of clusters
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.xticks(np.arange(1, 11))
plt.show()

# Determine the optimal number of clusters based on the elbow method
optimal_clusters = 3  # Adjust based on the elbow point on the plot

# Perform KMeans clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
clusters = kmeans.fit_predict(data_for_clustering)

# Add cluster labels to the merged data
merged_data['Cluster'] = clusters

# Plot clusters over the years
plt.figure(figsize=(12, 8))
for cluster in np.unique(clusters):
    cluster_data = merged_data[merged_data['Cluster'] == cluster]
    plt.plot(cluster_data['Date'], cluster_data['pollen_count'], label=f"Cluster {cluster}")
    
plt.xlabel("Date")
plt.ylabel("Pollen Count")
plt.title(f"Cluster Analysis of Pollen Count Patterns (Optimal Clusters: {optimal_clusters})")
plt.legend()
plt.tight_layout()
plt.show()


"""
Multivariate Analysis with Long-Term Data:
Apply multivariate analysis techniques to explore the relationships between weather, air quality, and pollen count across 
the entire 10-year dataset. Consider using principal component analysis (PCA) or factor analysis to identify dominant patterns and trends.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load merged dataset containing weather, air quality, and pollen count data
merged_data = pd.read_csv("pollen_count_dataset.csv")

# List of features for analysis
features_for_analysis = ["mean_temperature", "humidity", "wind_speed", "rain", "dew_point", "pressure", "pm25", "pm10", "no2", "o3", "pollen_count"]  # Include relevant features

# Prepare data for analysis
data_for_analysis = merged_data[features_for_analysis]

# Standardize the data
data_standardized = (data_for_analysis - data_for_analysis.mean()) / data_for_analysis.std()

# Apply PCA
pca = PCA()
principal_components = pca.fit_transform(data_standardized)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Plot explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(explained_variance_ratio))
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance Ratio")
plt.title("Explained Variance Ratio")
plt.show()

# Select number of components based on the explained variance ratio
num_components = 3  # Adjust as needed

# Reapply PCA with selected number of components
pca = PCA(n_components=num_components)
principal_components = pca.fit_transform(data_standardized)

# Add principal components to the merged data
for i in range(num_components):
    merged_data[f"PC{i+1}"] = principal_components[:, i]

# Scatterplot matrix of principal components
pd.plotting.scatter_matrix(merged_data[["PC1", "PC2", "PC3"]], figsize=(10, 8))
plt.suptitle("Scatterplot Matrix of Principal Components", y=1.02)
plt.show()

"""Weather Variability and Pollen Count:
Investigate how changes in weather conditions (e.g., rapid temperature fluctuations, sudden increases in humidity) might impact pollen count and if there are any relationships.

"""
import pandas as pd
import matplotlib.pyplot as plt

# Load merged dataset containing weather, air quality, and pollen count data
merged_data = pd.read_csv("pollen_count_dataset.csv")

# Convert the 'Date' column to datetime format
merged_data['Date'] = pd.to_datetime(merged_data['Date'])

# Calculate weather variability metrics
merged_data['temperature_variation'] = merged_data['max_temperature'] - merged_data['min_temperature']
merged_data['humidity_change'] = merged_data['humidity'].diff()
merged_data['wind_speed_change'] = merged_data['wind_speed'].diff()

# Calculate the change in pollen count from the previous day
merged_data['pollen_count_change'] = merged_data['pollen_count'].diff()

# Plot relationships between weather variability and pollen count change
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.scatter(merged_data['temperature_variation'], merged_data['pollen_count_change'], alpha=0.5)
plt.xlabel("Temperature Variation")
plt.ylabel("Change in Pollen Count")
plt.title("Temperature Variation vs. Pollen Count Change")

plt.subplot(2, 2, 2)
plt.scatter(merged_data['humidity_change'], merged_data['pollen_count_change'], alpha=0.5)
plt.xlabel("Humidity Change")
plt.ylabel("Change in Pollen Count")
plt.title("Humidity Change vs. Pollen Count Change")

plt.subplot(2, 2, 3)
plt.scatter(merged_data['wind_speed_change'], merged_data['pollen_count_change'], alpha=0.5)
plt.xlabel("Wind Speed Change")
plt.ylabel("Change in Pollen Count")
plt.title("Wind Speed Change vs. Pollen Count Change")

plt.tight_layout()
plt.show()

"""Comparative Analysis:
Compare pollen count patterns across different years or months to understand if certain years or periods consistently 
exhibit higher or lower pollen counts based on specific weather or air quality conditions.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load merged dataset containing weather, air quality, and pollen count data
merged_data = pd.read_csv("pollen_count_dataset.csv")

# Convert the 'Date' column to datetime format
merged_data['Date'] = pd.to_datetime(merged_data['Date'])

# Filter data for the spring and summer months
start_date = pd.Timestamp(year=merged_data['Date'].min().year, month=4, day=20)
end_date = pd.Timestamp(year=merged_data['Date'].max().year, month=9, day=10)
filtered_data = merged_data[(merged_data['Date'] >= start_date) & (merged_data['Date'] <= end_date)]

# Select relevant weather features for comparison
weather_features = ["mean_temperature", "humidity", "wind_speed"]  # Include relevant features

# Create subplots to compare pollen count patterns across different weather conditions
plt.figure(figsize=(12, 8))
for i, feature in enumerate(weather_features, 1):
    plt.subplot(1, len(weather_features), i)
    sns.boxplot(x=filtered_data[feature], y=filtered_data['pollen_count'])
    plt.xlabel(feature)
    plt.ylabel("Pollen Count")
    plt.title(f"Comparative Analysis by {feature.capitalize()}")

plt.tight_layout()
plt.show()

"""Interaction Effects:
Explore interactions between multiple weather features and air quality features to see if certain combinations result in more pronounced changes in pollen count.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load merged dataset containing weather, air quality, and pollen count data
merged_data = pd.read_csv("pollen_count_dataset.csv")

# Select relevant features for interaction analysis
selected_features = ["mean_temperature", "humidity", "wind_speed", "rain", "dew_point", "pressure", "pm25", "pm10", "no2", "o3", "pollen_count"]

# Create a scatterplot matrix (pair plot) to visualize interactions
sns.set(style="ticks")
sns.pairplot(merged_data[selected_features])
plt.suptitle("Scatterplot Matrix of Weather, Air Quality, and Pollen Count Features", y=1.02)
plt.show()