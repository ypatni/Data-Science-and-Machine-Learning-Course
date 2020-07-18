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
#making a heatmap 
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