# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Load datasets
medal_data = pd.read_csv("summerOly_medal_counts.csv")  # Historical Olympic medal data
sport_data = pd.read_csv("summary.csv")  # Detailed sports data

# Display the first few rows of the datasets for verification
print("Medal Data:")
print(medal_data.head())
print("Sport Data:")
print(sport_data.head())

# 2. Data Preprocessing
# Ensure critical columns are numeric
medal_data['Year'] = pd.to_numeric(medal_data['Year'], errors='coerce')
medal_data['Gold'] = pd.to_numeric(medal_data['Gold'], errors='coerce')
medal_data['Total'] = pd.to_numeric(medal_data['Total'], errors='coerce')

# Drop rows with missing values in essential columns
medal_data.dropna(subset=['Year', 'Gold', 'Total'], inplace=True)

# 3. Feature Engineering
# 3.1 Calculate weighted historical performance for each country
def calculate_weighted_average(data, years, medal_col, decay=0.1):
    weights = np.exp(-decay * (max(years) - years))
    weighted_average = (data[medal_col] * weights).sum() / weights.sum()
    return weighted_average

# Compute weighted averages for total medals and gold medals
historical_medal_avg = []
for noc in medal_data['NOC'].unique():
    noc_data = medal_data[medal_data['NOC'] == noc]
    total_avg = calculate_weighted_average(noc_data, noc_data['Year'], 'Total')
    gold_avg = calculate_weighted_average(noc_data, noc_data['Year'], 'Gold')
    historical_medal_avg.append({'NOC': noc, 'Weighted_Total': total_avg, 'Weighted_Gold': gold_avg})

weighted_medal_df = pd.DataFrame(historical_medal_avg)

# 3.2 Calculate project efficiency
sport_data['Total_Medals'] = sport_data['Gold_Count'] + sport_data['Silver_Count'] + sport_data['Bronze_Count']
project_efficiency = sport_data.groupby(['Team'])['Total_Medals'].sum().reset_index()
project_efficiency.rename(columns={'Total_Medals': 'Project_Efficiency', 'Team': 'NOC'}, inplace=True)

# 3.3 Merge historical performance and project efficiency
final_data = weighted_medal_df.merge(project_efficiency, on='NOC', how='left')
final_data.fillna(0, inplace=True)

# Display the final processed data
print("Final Data:")
print(final_data.head())

# 4. Model Training and Prediction
# 4.1 Total Medals Prediction
X_total = final_data[['Weighted_Total', 'Project_Efficiency']]
y_total = final_data['Weighted_Total']

# Standardize features
scaler = StandardScaler()
X_total_scaled = scaler.fit_transform(X_total)

# Train-test split
X_train_total, X_test_total, y_train_total, y_test_total = train_test_split(X_total_scaled, y_total, test_size=0.2, random_state=42)

# Train a linear regression model
reg_total = LinearRegression()
reg_total.fit(X_train_total, y_train_total)

# Predict total medals
y_pred_total = reg_total.predict(X_test_total)

# Evaluate the total medals model
print("Total Medals Prediction Model:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test_total, y_pred_total))
print("R-squared (R²):", r2_score(y_test_total, y_pred_total))

# 4.2 Gold Medals Prediction
X_gold = final_data[['Weighted_Gold', 'Project_Efficiency']]
y_gold = final_data['Weighted_Gold']

# Standardize features
X_gold_scaled = scaler.fit_transform(X_gold)

# Train-test split
X_train_gold, X_test_gold, y_train_gold, y_test_gold = train_test_split(X_gold_scaled, y_gold, test_size=0.2, random_state=42)

# Train a linear regression model
reg_gold = LinearRegression()
reg_gold.fit(X_train_gold, y_train_gold)

# Predict gold medals
y_pred_gold = reg_gold.predict(X_test_gold)

# Evaluate the gold medals model
print("\nGold Medals Prediction Model:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test_gold, y_pred_gold))
print("R-squared (R²):", r2_score(y_test_gold, y_pred_gold))

# 5. Generate Predictions for All Countries
# Predict total medals
final_data['Predicted_Total'] = reg_total.predict(scaler.transform(final_data[['Weighted_Total', 'Project_Efficiency']]))

# Predict gold medals
final_data['Predicted_Gold'] = reg_gold.predict(scaler.transform(final_data[['Weighted_Gold', 'Project_Efficiency']]))

# Display predictions
print("\nPrediction Results:")
print(final_data[['NOC', 'Predicted_Total', 'Predicted_Gold']])

# 6. Visualization
# Total Medals Prediction
plt.figure(figsize=(8, 6))
plt.scatter(y_test_total, y_pred_total, alpha=0.7, label='Predicted vs Actual')
plt.plot([min(y_test_total), max(y_test_total)], [min(y_test_total), max(y_test_total)], 'r--', label='Perfect Prediction')
plt.xlabel("Actual Total Medals")
plt.ylabel("Predicted Total Medals")
plt.title("Total Medals Prediction")
plt.legend()
plt.show()

# Gold Medals Prediction
plt.figure(figsize=(8, 6))
plt.scatter(y_test_gold, y_pred_gold, alpha=0.7, color='orange', label='Predicted vs Actual')
plt.plot([min(y_test_gold), max(y_test_gold)], [min(y_test_gold), max(y_test_gold)], 'r--', label='Perfect Prediction')
plt.xlabel("Actual Gold Medals")
plt.ylabel("Predicted Gold Medals")
plt.title("Gold Medals Prediction")
plt.legend()
plt.show()
