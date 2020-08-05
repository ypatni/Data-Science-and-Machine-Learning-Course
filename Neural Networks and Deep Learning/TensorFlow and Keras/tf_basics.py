import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

mpl.rcParams['patch.force_edgecolor'] = True
sns.set()
sns.set_style('whitegrid')
#very fake dataset

df = pd.read_csv('~/Desktop/Code/Udemy DS + ML/Neural Networks and Deep Learning/Tensorflow/DATA/fake_reg.csv')
print(df.head())
sns.pairplot(df)
plt.show()

#test train split 

X = df[['feature1', 'feature2']].values #we have to use numpy arrays 
y = df['price'].values 
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)
print(X_train.shape)

#normalizing and scaling feature data 
#help(MinMaxScaler)
scaler = MinMaxScaler()
scaler.fit(X_train) #calculates params needed to perform the actual scaling 
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.max())



