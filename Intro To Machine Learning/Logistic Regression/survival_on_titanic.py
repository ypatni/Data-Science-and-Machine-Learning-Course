import pandas as  pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
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
sns.heatmap(train.isnull(), yticklabels=False, cbar = False, cmap= 'viridis')
plt.show() #now no missing age columns 

