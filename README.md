# PRODIGY_DS_02
data cleaning and exploratory data analysis (EDA)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


titanic_df = pd.read_csv(r"C:\Users\Admin\Downloads\titanic.csv.csv")

titanic_df.head()


titanic_df.isnull().sum()


titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)


titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0], inplace=True)

sns.barplot(x='Sex', y='Survived', data=titanic_df, ci=None)
plt.title('Survival Rate by Gender')
plt.show()

sns.barplot(x='Pclass', y='Survived', data=titanic_df, ci=None)
plt.title('Survival Rate by Passenger Class')
plt.show()

plt.hist(titanic_df['Age'], bins=20, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution')
plt.show()

corr = titanic_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
