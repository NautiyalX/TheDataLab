import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Generate synthetic weather data
np.random.seed(42)
days = np.arange(1, 366)
temperature = np.random.normal(loc=20, scale=5, size=365)
humidity = np.random.normal(loc=60, scale=15, size=365)
wind_speed = np.random.normal(loc=10, scale=3, size=365)

# 2. Basic statistics for temperature
mean_temp = np.mean(temperature)
median_temp = np.median(temperature)
var_temp = np.var(temperature)
min_temp = np.min(temperature)
max_temp = np.max(temperature)

print(f"Temperature Stats -> Mean: {mean_temp:.2f}, Median: {median_temp:.2f}, Variance: {var_temp:.2f}")
print(f"Min: {min_temp:.2f}, Max: {max_temp:.2f}")

# 3. Detect anomalies using IQR method
Q1 = np.percentile(temperature, 25)
Q3 = np.percentile(temperature, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

anomalies_idx = np.where((temperature < lower_bound) | (temperature > upper_bound))[0]
anomalies_vals = temperature[anomalies_idx]

print(f"Detected {len(anomalies_idx)} temperature anomalies at days: {anomalies_idx + 1}")

# 4. Visualization
plt.figure(figsize=(12, 6))
plt.plot(days, temperature, label='Temperature (°C)')
plt.scatter(anomalies_idx + 1, anomalies_vals, color='red', label='Anomalies', s=50, marker='x')
plt.xlabel('Day of Year')
plt.ylabel('Temperature (°C)')
plt.title('Daily Temperature with Anomalies Highlighted')
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
sns.histplot(temperature, kde=True, color='skyblue')
plt.title('Temperature Distribution with KDE')
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x=temperature, color='lightgreen')
plt.title('Boxplot of Temperature')
plt.show()
