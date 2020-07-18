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
sns.lmplot(x = "Length of Membership", y= "Yearly Amount Spent", data = df)
plt.show()
#setting training and testing data 
print(df.columns)
y= df["Yearly Amount Spent"] #what were trying to predict 
x = df[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

lm = LinearRegression()

lm.fit(x_train,y_train)
#printing coefficients

print(lm.coef_)

#predicting the testing data 
predictions = lm.predict(x_test)

plt.scatter(y_test, predictions)
plt.xlabel('Y Test - True Values')
plt.ylabel('Predicted Values')
plt.show()

#evaluating model
print(f"\n")
print('MAE', metrics.mean_absolute_error(y_test, predictions))
print(f"\n")
print('MSE',metrics.mean_squared_error(y_test, predictions))
print(f"\n")
print('RMSE',np.sqrt(metrics.mean_squared_error(y_test, predictions))) 
print(f"\n")

#plotting residuals

sns.distplot((y_test-predictions), bins=30)
plt.show()

cdf = pd.DataFrame(lm.coef_, x.columns, columns=['Coeff'])
print(cdf)