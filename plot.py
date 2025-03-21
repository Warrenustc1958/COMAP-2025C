import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
file_path = 'summerOly_medal_counts.csv'  # 替换为实际文件路径
data = pd.read_csv(file_path)

# 筛选出 United States 和 China 的数据
countries = ['United States', 'China']  # 替换为实际国家名称
filtered_data = data[data['NOC'].isin(countries)]

# 绘制 United States 的数据随时间变化
us_data = filtered_data[filtered_data['NOC'] == 'United States']
plt.figure(figsize=(10, 6))
plt.bar(us_data['Year'], us_data['Gold'], label='Gold', alpha=0.7)
plt.bar(us_data['Year'], us_data['Silver'], label='Silver', alpha=0.7, bottom=us_data['Gold'])
plt.bar(us_data['Year'], us_data['Bronze'], label='Bronze', alpha=0.7, bottom=us_data['Gold'] + us_data['Silver'])
plt.title('United States Medal Count Over Time')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend()
plt.show()

# 绘制 China 的数据随时间变化
china_data = filtered_data[filtered_data['NOC'] == 'China']
plt.figure(figsize=(10, 6))
plt.bar(china_data['Year'], china_data['Gold'], label='Gold', alpha=0.7)
plt.bar(china_data['Year'], china_data['Silver'], label='Silver', alpha=0.7, bottom=china_data['Gold'])
plt.bar(china_data['Year'], china_data['Bronze'], label='Bronze', alpha=0.7, bottom=china_data['Gold'] + china_data['Silver'])
plt.title('China Medal Count Over Time')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend()
plt.show()

