from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = load_iris(as_frame=True)
df = iris.frame

sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='target', palette='Set1')
plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()

sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', hue='target', palette='Set2')
plt.title('Petal Length vs Petal Width')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.show()

sns.boxplot(data=df, x='target', y='sepal length (cm)', palette='pastel')
plt.title('Sepal Length by Species')
plt.xlabel('Species')
plt.ylabel('Sepal Length (cm)')
plt.show()

sns.boxplot(data=df, x='target', y='petal length (cm)', palette='pastel')
plt.title('Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()

sns.heatmap(df.drop(columns='target').corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
