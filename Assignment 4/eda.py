import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = sns.load_dataset('titanic')
print(df.info())
print(df.isnull().sum())
print(df.describe(include='all'))

plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cmap='viridis', cbar=False)
plt.title('Missing Values in Dataset')
plt.show()

num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols].hist(bins=20, figsize=(12, 8), color='skyblue', edgecolor='black')
plt.suptitle('Histograms of Numeric Features', y=1.02)
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(x='class', data=df, order=df['class'].value_counts())
plt.title('Count of Class')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='class', y='age', data=df, palette='Set2')
plt.title('Age Distribution by Class')
plt.show()

sns.pairplot(df.dropna(subset=['age']), hue='survived')
plt.show()

plt.figure(figsize=(10, 6))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x='embark_town', hue='survived', data=df)
plt.title('Survival by Embarkation Town')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(y='fare', data=df, color='lightgreen')
plt.title('Boxplot of Fare')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['age'].dropna(), bins=20, color='orchid')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
