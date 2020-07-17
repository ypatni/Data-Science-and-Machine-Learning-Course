import numpy as np 
import pandas as pd 
#vuisualization imports and setup
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import chart_studio.plotly as py 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston #for real dataset 
from sklearn import metrics 
mpl.rcParams['patch.force_edgecolor'] = True
sns.set()

df = pd.read_csv('Ecommerce Customers')
print(df.head())
print(df.info())
print(df.describe())

sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=df)
plt.show()
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=df)
plt.show()

