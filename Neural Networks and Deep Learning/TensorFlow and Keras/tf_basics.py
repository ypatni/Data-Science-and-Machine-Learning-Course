import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
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


model.fit(x=X_train, y = y_train, epochs=250) # 1 epoch = 1 pass over the entire dataset

loss_df = pd.DataFrame(model.history.history)
print(loss_df)
loss_df.plot()
plt.show()

#model evaluation 
print(model.evaluate(X_test, y_test, verbose=0)) #this gives back the mean squared error
print(model.evaluate(X_train, y_train, verbose=0))

test_predictions = model.predict(X_test)
test_predictions = pd.Series(test_predictions.reshape(300,))
pred_df = pd.DataFrame(y_test, columns=['test true y'])
pred_df = pd.concat([pred_df, test_predictions], axis = 1)
pred_df.columns = ['test true y', 'model predictions']
print(pred_df)

sns.scatterplot(x='test true y', y = 'model predictions', data=pred_df)
plt.show()
print(mean_absolute_error(pred_df['test true y'], pred_df['model predictions']))
print(mean_squared_error(pred_df['test true y'], pred_df['model predictions']))

new_gem = [[998, 1000]]

new_gem = scaler.transform(new_gem)
print(model.predict(new_gem)) #price estimated to be around $420
model.save('my_gem_model.h5')

later_model = load_model('my_gem_model.h5')
print(later_model.predict(new_gem)) #saving and loading model 