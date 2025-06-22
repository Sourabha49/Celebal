import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

california = fetch_california_housing(as_frame=True)
df = california.frame

print(df.head())

plt.figure(figsize=(8, 5))
sns.histplot(df['MedHouseVal'], kde=True, color='skyblue')
plt.title("Distribution of Median House Prices")
plt.xlabel("Median House Value (in $100,000s)")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x='MedInc', y='MedHouseVal', data=df, alpha=0.5)
plt.title("House Price vs Median Income")
plt.xlabel("Median Income (in $10,000s)")
plt.ylabel("Median House Value (in $100,000s)")
plt.show()

plt.figure(figsize=(10, 6))
sc = plt.scatter(x=df['Longitude'], y=df['Latitude'], c=df['MedHouseVal'], cmap='viridis', s=15, alpha=0.6)
plt.colorbar(sc, label="Median House Value (in $100,000s)")
plt.title("California Housing Prices by Location")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x=pd.cut(df['HouseAge'], bins=[0, 10, 20, 30, 40, 50]), y='MedHouseVal', data=df)
plt.title("House Price by Age Groups")
plt.xlabel("House Age (binned)")
plt.ylabel("Median House Value")
plt.xticks(rotation=45)
plt.show()
