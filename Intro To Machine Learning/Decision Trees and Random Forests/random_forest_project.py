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

loans = pd.read_csv('loan_data.csv')
print(loans.head())
print(loans.info())

plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')
plt.show()
plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',bins=30,label='Not Fully Paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',bins=30,label='Not Fully Paid=0')
plt.legend()
plt.xlabel('FICO')
plt.show()
plt.figure(figsize=(11,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')
plt.show()
sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')
plt.show()

cat_feats = ['purpose']
final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True) 

x = final_data.drop('not.fully.paid', axis = 1)
y = final_data['not.fully.paid']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 101)
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
