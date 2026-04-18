import pandas as pd
df = pd.read_csv("dynamic_supply_chain_logistics_dataset_with_country.csv")
print(df.head())
print(df.info())
print(df.describe())
import matplotlib.pyplot as plt
import seaborn as sns
sns.histplot(df['delay_probability'])
plt.show()
sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), cmap='coolwarm')
plt.show()
sns.boxplot(x='risk_classification', y='delay_probability', data=df)
plt.show()
sns.scatterplot(x='supplier_reliability_score', y='delay_probability', data=df)
plt.show()
df['risk_bins'] = pd.cut(df['route_risk_level'], bins=5)
sns.boxplot(x='risk_bins', y='delay_probability', data=df)
plt.xticks(rotation=45)
plt.show()
sns.boxplot(x='supplier_country', y='delay_probability', data=df)
plt.xticks(rotation=90)
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X = df.select_dtypes(include=['float64']).drop(columns=['delay_probability'])
y = df['delay_probability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))