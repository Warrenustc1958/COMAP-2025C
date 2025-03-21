# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay
# 新增国家名称标准化函数
def standardize_country_names(df):
    """将国家名称标准化以匹配地理数据"""
    country_mapping = {
        'United States': 'United States of America',
        'Great Britain': 'United Kingdom',
        'Russia': 'Russian Federation',
        'South Korea': 'South Korea',
        'North Korea': 'North Korea',
        'Czech Republic': 'Czechia',
        'Taiwan': 'Taiwan',
        'Serbia': 'Republic of Serbia',
        'Macedonia': 'North Macedonia',
        'Congo': 'Democratic Republic of the Congo'
    }
    df['Country'] = df['Team'].replace(country_mapping)
    return df
# 1. Load data
sport_data = pd.read_csv("summary.csv")

# 2. Preprocess data
# Calculate total medals for each entry
sport_data['Total_Medals'] = sport_data['Gold_Count'] + sport_data['Silver_Count'] + sport_data['Bronze_Count']

# Aggregate data by country
country_summary = sport_data.groupby('Team').agg(
    Total_Medals=('Total_Medals', 'sum'),
    Num_Sports=('Sport', 'nunique')
).reset_index()

# Determine if a country has ever won a medal
country_summary['Ever_Won_Medals'] = (country_summary['Total_Medals'] > 0).astype(int)

# 3. Prepare features and labels
X = country_summary[['Num_Sports']]
y = country_summary['Ever_Won_Medals']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train Logistic Regression model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=40)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
y_pred_prob = log_reg.predict_proba(X_test)[:, 1]

# 5. Evaluate the model
print("Logistic Regression Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_prob))

# 6. Predict for countries without medals
unawarded_countries = country_summary[country_summary['Ever_Won_Medals'] == 0].copy()
unawarded_countries['Medal_Prob'] = log_reg.predict_proba(
    scaler.transform(unawarded_countries[['Num_Sports']])
)[:, 1]

# Sort by probability
unawarded_countries = unawarded_countries.sort_values(by='Medal_Prob', ascending=False)
print("\nTop 10 countries predicted to win their first medal:")
print(unawarded_countries[['Team', 'Medal_Prob']].head(10))

# Predict number of first-time medal-winning countries
threshold = 0.5
unawarded_countries['Will_Win_Medal'] = (unawarded_countries['Medal_Prob'] >= threshold).astype(int)
num_first_medals = unawarded_countries['Will_Win_Medal'].sum()
print(f"\nNumber of countries predicted to win their first medal: {num_first_medals}")

# 7. Visualize probability distribution
plt.figure(figsize=(10, 6))
sns.histplot(unawarded_countries['Medal_Prob'], bins=20, kde=True, color='blue')
plt.title("Predicted Medal Probabilities for Countries Without Medals")
plt.xlabel("Medal Probability")
plt.ylabel("Number of Countries")
plt.show()

# 8. Display countries predicted to win their first medal
print("\nCountries predicted to win their first medal:")
print(unawarded_countries[unawarded_countries['Will_Win_Medal'] == 1][['Team', 'Medal_Prob']])

# 9. Visualize Logistic Regression Coefficients (Feature Importance)
plt.figure(figsize=(8, 6))
coefficients = log_reg.coef_[0]
plt.barh(['Num_Sports'], coefficients, color='green')
plt.xlabel("Logistic Regression Coefficients")
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.show()

# 7. 增强可视化（修改原有直方图并新增图表）
plt.figure(figsize=(18, 12))

# 子图1：概率分布直方图
plt.subplot(2, 2, 1)
sns.histplot(unawarded_countries['Medal_Prob'], bins=20, kde=True, color='red')
plt.title("Predicted Medal Probabilities Distribution")
plt.xlabel("Medal Probability")
plt.ylabel("Number of Countries")
plt.axvline(threshold, color='blue', linestyle='--', label=f'Threshold ({threshold})')
plt.legend()

# 子图2：Top15国家预测概率
plt.subplot(2, 2, 2)
top_countries = unawarded_countries.nlargest(15, 'Medal_Prob')
sns.barplot(x='Medal_Prob', y='Team', data=top_countries, palette='cividis')
plt.title('Top 15 Unawarded Countries by Medal Probability')
plt.xlabel('Probability')
plt.ylabel('Team')
plt.xlim(0, 1)

# 子图3：运动项目数量与获奖概率关系
plt.subplot(2, 2, 3)
sns.scatterplot(x='Num_Sports', y='Medal_Prob', data=unawarded_countries, 
                hue='Medal_Prob', palette='coolwarm', size='Num_Sports')
plt.title('Number of Sports vs. Medal Probability')
plt.xlabel('Number of Sports Participated')
plt.ylabel('Predicted Probability')
plt.axhline(threshold, color='red', linestyle='--')

# 子图4：预测结果分布
plt.subplot(2, 2, 4)
sns.countplot(x='Will_Win_Medal', data=unawarded_countries, palette='Set2')
plt.title('Predicted Medal Winners Distribution')
plt.xlabel('Will Win Medal (1=Yes, 0=No)')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])

plt.tight_layout()
plt.show()

# 新增可视化：混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Medal', 'Has Medal'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()

# 5. Evaluate the model（在原有评估代码后添加可视化）
print("\nLogistic Regression Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_prob))

# 新增可视化1：ROC曲线
plt.figure(figsize=(10, 6))
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_pred_prob):.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# 新增可视化2：特征重要性分析（逻辑回归系数）
plt.figure(figsize=(10, 6))
importance = log_reg.coef_[0]
feature_names = ['Number of Sports']
sorted_idx = np.argsort(np.abs(importance))
plt.barh(range(len(sorted_idx)), np.abs(importance[sorted_idx]), align='center', color='royalblue')
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel('Absolute Coefficient Value')
plt.title('Feature Importance Analysis')
plt.tight_layout()
plt.show()