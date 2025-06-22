from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = load_iris(as_frame=True)
df = iris.frame
df['species'] = df['target'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

sns.pairplot(df, hue='species')
plt.show()

sns.countplot(data=df, x='species')
plt.title('Number of Flowers per Species')
plt.show()

sns.violinplot(data=df, x='species', y='sepal length (cm)')
plt.title('Sepal Length by Species')
plt.show()

sns.violinplot(data=df, x='species', y='petal length (cm)')
plt.title('Petal Length by Species')
plt.show()

sns.boxplot(data=df, x='species', y='sepal width (cm)')
plt.title('Sepal Width by Species')
plt.show()

sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', hue='species')
plt.title('Petal Length vs Petal Width')
plt.show()

sns.heatmap(df.drop(columns=['target','species']).corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()
