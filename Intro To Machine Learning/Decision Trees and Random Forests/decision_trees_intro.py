import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

mpl.rcParams['patch.force_edgecolor'] = True
sns.set()
sns.set_style('whitegrid')

df = pd.read_csv('kyphosis.csv')
print(df.head())

sns.pairplot(df, hue = 'Kyphosis')
plt.show()

x = df.drop('Kyphosis', axis = 1)
y = df['Kyphosis']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

dtree = DecisionTreeClassifier()
dtree.fit(x_train, y_train)
predictions = dtree.predict(x_test)
print(classification_report(y_test, predictions)) 
print(confusion_matrix(y_test, predictions)) 

rfc = RandomForestClassifier(n_estimators = 200)
rfc.fit(x_train, y_train)
rfc_pred = rfc.predict(x_test)
print(classification_report(y_test, rfc_pred)) 
print(confusion_matrix(y_test, rfc_pred)) 