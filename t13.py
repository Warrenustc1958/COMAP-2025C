import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
medal_data = pd.read_csv("summerOly_medal_counts.csv")
sport_data = pd.read_csv("summary.csv")

# Ensure necessary columns are numeric
sport_data['Gold_Count'] = pd.to_numeric(sport_data['Gold_Count'], errors='coerce')
sport_data['Silver_Count'] = pd.to_numeric(sport_data['Silver_Count'], errors='coerce')
sport_data['Bronze_Count'] = pd.to_numeric(sport_data['Bronze_Count'], errors='coerce')

# Drop rows with missing values in critical columns
sport_data.dropna(subset=['Gold_Count', 'Silver_Count', 'Bronze_Count'], inplace=True)

# Add a total medals column for sport_data
sport_data['Total_Medals'] = sport_data['Gold_Count'] + sport_data['Silver_Count'] + sport_data['Bronze_Count']

# 2. Analyze the relationship between events and medals
# Group by sport and sum the total medals
sport_medals = sport_data.groupby('Sport')['Total_Medals'].sum().reset_index()

# Sort by the total medals
sport_medals = sport_medals.sort_values(by='Total_Medals', ascending=False)

# Plot the total medals by sport
plt.figure(figsize=(12, 6))
sns.barplot(data=sport_medals.head(20), x='Total_Medals', y='Sport', palette='viridis')
plt.title('Top 20 Sports by Total Medals')
plt.xlabel('Total Medals')
plt.ylabel('Sport')
plt.show()

# 3. Determine key sports for various countries
# Group by Team and Sport to see medal distribution
country_sport_medals = sport_data.groupby(['Team', 'Sport'])['Total_Medals'].sum().reset_index()

# Find the top sports for each country
top_sports_by_country = country_sport_medals.sort_values(['Team', 'Total_Medals'], ascending=[True, False]).groupby('Team').head(3)

print("Top Sports by Country:")
print(top_sports_by_country.head(20))

# Plot the top sports for a specific country (e.g., United States)
country = "United States"
country_data = country_sport_medals[country_sport_medals['Team'] == country].sort_values(by='Total_Medals', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=country_data.head(10), x='Total_Medals', y='Sport', palette='coolwarm')
plt.title(f'Top Sports for {country}')
plt.xlabel('Total Medals')
plt.ylabel('Sport')
plt.show()

# 4. Home Country Impact Analysis
# Replace 'Host_Country_NOC' with the actual host country NOC (e.g., 'USA')
host_noc = "USA"  # Example host country NOC
medal_data['Is_Host'] = medal_data['NOC'] == host_noc

# Debug: Check if host rows are identified
host_country = medal_data[medal_data['Is_Host'] == True]
print("Host Country Data:")
print(host_country.head())
print("Number of rows for the host country:", len(host_country))

# Merge with sport_data
host_sport_data = pd.merge(host_country, sport_data, left_on=['NOC', 'Year'], right_on=['Team', 'Year'], how='inner')
print("Host Sport Data (after merge):")
print(host_sport_data.head())
print("Number of rows in host_sport_data:", len(host_sport_data))

# Group by sport
host_sport_performance = host_sport_data.groupby('Sport')['Total_Medals'].sum().reset_index()
print("Host Country Sport Performance:")
print(host_sport_performance.head())
print("Number of rows in host_sport_performance:", len(host_sport_performance))

# Plot only if data is available
if not host_sport_performance.empty:
    plt.figure(figsize=(12, 6))
    sns.barplot(data=host_sport_performance.sort_values(by='Total_Medals', ascending=False), 
                x='Total_Medals', y='Sport', palette='mako')
    plt.title('Host Country Performance by Sport')
    plt.xlabel('Total Medals')
    plt.ylabel('Sport')
    plt.show()
else:
    print("No data available for host country performance by sport.")
