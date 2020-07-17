import numpy as np 
import pandas as pd 
#vuisualization imports and setup
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import chart_studio.plotly as py 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

x = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']

#splitting data into train, test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)
# we basically use tuple unpacking to get your training and testing variables, test size is the percentage of data set allocated to the test size, 
# random state ensures a specific set of random splits on data 

lm = LinearRegression()
lm.fit(x_train, y_train)
print(f"\n")
print(lm.intercept_)
print(f"\n")
print(lm.coef_) #returning coefficients the coefficients relate to the columns in x_train
cdf = pd.DataFrame(lm.coef_, x.columns, columns=['Coef']) #making a dataframe using the coefficients so that we can visualize and use them
print(cdf)

