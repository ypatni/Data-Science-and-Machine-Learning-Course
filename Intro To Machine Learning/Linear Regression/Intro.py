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
# so if we hold all other features fixed, a one unit increase in average income is associated with an increase in $21.53 in price
# the data we have is artificial and makes literally no sense
 
#CREATING PREDICTIONS FROM TEST SET 

predictions = lm.predict(x_test)
print(predictions) #creates array of predicted prices of house 
#but we know y test has the correct prices of the houses, so we want to know how far off the predictions are from the actual values 
#so we make a scatter plot!
plt.scatter(y_test, predictions)
plt.show()
#we get a pretty staright line which is very good 
#now we create a histogram of the residuals

sns.distplot((y_test-predictions))
plt.show()

