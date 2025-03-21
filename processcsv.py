import pandas as pd

# 读取 CSV 文件
file_path = 'summerOly_athletes.csv'  # 替换为你的文件路径
data = pd.read_csv(file_path)

# 创建奖牌类型的标记列
data['Gold'] = data['Medal'].apply(lambda x: 1 if x == 'Gold' else 0)
data['Silver'] = data['Medal'].apply(lambda x: 1 if x == 'Silver' else 0)
data['Bronze'] = data['Medal'].apply(lambda x: 1 if x == 'Bronze' else 0)

# 按 Team, Year, Sport 分组，统计每种奖牌的数量
medal_stats = data.groupby(['Team', 'Year', 'Sport']).agg(
    Gold_Count=('Gold', 'sum'),
    Silver_Count=('Silver', 'sum'),
    Bronze_Count=('Bronze', 'sum')
).reset_index()

# 保存结果到新文件
medal_stats.to_csv('medal_stats_summary.csv', index=False)

# 打印结果
print(medal_stats)
