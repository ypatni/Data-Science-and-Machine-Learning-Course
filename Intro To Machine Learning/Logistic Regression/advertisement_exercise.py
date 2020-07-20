import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
mpl.rcParams['patch.force_edgecolor'] = True
sns.set()
ad_data = pd.read_csv('advertising.csv')
print(ad_data.head())
print(ad_data.info())
sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
plt.show()
#Area Income vs Age
sns.jointplot(x = 'Age', y='Area Income', data = ad_data)
plt.show()
#time spent on site vs age 
sns.jointplot(x = 'Age', y='Daily Time Spent on Site', data = ad_data, color = 'green', kind = 'kde')
plt.show()
#sns.pairplot(ad_data,hue='Clicked on Ad',palette='bwr')
#plt.show() - very laggy 

#data we can't really use in a logistic regression - timestamp, topic line, city, country
ad_data.drop(['Timestamp', 'Ad Topic Line', 'City','Country'],axis = 1, inplace = True)
print(ad_data.info())

x = ad_data.drop('Clicked on Ad', axis = 1)
y = ad_data['Clicked on Ad']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)
predictions = logmodel.predict(x_test)
print(classification_report(y_test, predictions)) 
print(confusion_matrix(y_test, predictions)) 
