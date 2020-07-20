import pandas as  pd 
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

train = pd.read_csv('titanic_train.csv')
print(train.head())
print(train.isnull()) 
#making a heatmap to figure out missing data 
sns.heatmap(train.isnull(), yticklabels=False, cbar = False, cmap= 'viridis')
plt.show() #missing data in age column, cabin column and embarked 

sns.set_style('whitegrid')
sns.countplot(x = 'Survived', hue='Sex',  data=train, palette = 'RdBu_r')
plt.show()
sns.countplot(x = 'Survived', hue='Pclass',  data=train)
plt.show()
sns.distplot(train['Age'].dropna(), kde=False, bins = 30)
#train['Age'].plot.hist()
plt.show()
sns.countplot(x = 'SibSp', data = train)
plt.show()
train['Fare'].plot.hist(bins=40, figsize= (10,4))
plt.show()

#cleaning data for ML algorithms 
# using average age by passenger class in order to fill in missing age data (imputation)
plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass', y = 'Age', data = train)
plt.show() #shows that wealthier classes have older people 
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass ==2:
            return 29
        else:
            return 24
    else: 
        return Age
train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis = 1)
#checking again for missing age data 
train.drop('Cabin', axis = 1, inplace = True)
train.dropna(inplace=True)
sns.heatmap(train.isnull(), yticklabels=False, cbar = False, cmap= 'viridis')#now no missing  columns
#plt.show()  

sex = pd.get_dummies(train['Sex'], drop_first = True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)

train = pd.concat([train, sex, embark], axis = 1)
train.drop(['Sex', 'Embarked', 'Name','Ticket', 'PassengerId'],axis = 1, inplace = True)
print(train.head()) #all the data is now numeical and can be used by the ML algorithm 
 
x = train.drop('Survived', axis = 1)
y = train['Survived']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)
predictions = logmodel.predict(x_test)

print(classification_report(y_test, predictions)) 

#when using the confusion matrix  - print(confusion_matrix(y_test, predictions)) 
