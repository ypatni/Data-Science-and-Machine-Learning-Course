import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf 
mpl.rcParams['patch.force_edgecolor'] = True
sns.set()
sns.set_style('whitegrid')
#very fake dataset

df = pd.read_csv('~/Desktop/Code/Udemy DS + ML/Neural Networks and Deep Learning/Tensorflow and Keras/DATA/fake_reg.csv')
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

#creating and training a keras based model 

# help(Sequential) - really good documentation 
# model = Sequential([Dense(4, activation='relu'),Dense(2, activation='relu'), Dense(1)])

#adding layers one at a time makes it easier to edit 
model = Sequential()
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1)) #we only want to predict one value, the price

model.compile(optimizer='rmsprop', loss= 'mse') #loss depends on what type of task you try to solve



