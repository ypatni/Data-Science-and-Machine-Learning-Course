import numpy as np 
import pandas as pd 
#vuisualization imports and setup
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import chart_studio.plotly as py 
mpl.rcParams['patch.force_edgecolor'] = True
sns.set()


df = pd.read_csv('USA_Housing.csv')
print(df.head())
print(df.info())
print(df.describe())#gives statistical info aout columns with integer values 
print(df.columns) #- gives the names of columns

#sns.pairplot(df)
print(df.corr()) # shows correlation between columns 
sns.heatmap(df.corr(), annot=True, cmap = 'coolwarm')
plt.show()
sns.distplot(df['Price'])
plt.show()

x = df['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']
y = df['Price']