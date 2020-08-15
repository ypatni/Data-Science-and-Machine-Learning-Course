import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import tensorflow as tf 
mpl.rcParams['patch.force_edgecolor'] = True
sns.set()
sns.set_style('whitegrid')

#real world data

df = pd.read_csv('DATA/kc_house_data.csv')
print(df.head())
#print(df.isnull().sum()) #no missing data! 
#print(df.describe().transpose())
plt.figure(figsize = (10,6))
sns.distplot(df['price'])
plt.show()
sns.countplot(df['bedrooms'])
plt.show()
print(df.corr()['price'].sort_values()) # - checking for correlations
plt.figure(figsize = (10,6))
sns.scatterplot(x = 'price', y = 'sqft_living', data = df )
plt.show()
plt.figure(figsize = (10,6))
sns.boxplot(x='bedrooms', y = 'price', data=df)
plt.show()
plt.figure(figsize = (12,9))
sns.scatterplot(x='price', y='long', data =df)
plt.show()
plt.figure(figsize = (12,9))
sns.scatterplot(x='price', y='lat', data =df)
plt.show()
#at a certain value of latitiude and longitude, the price is very expensive 
bottom_ninety_nine_perc = df.sort_values('price', ascending=False).iloc[216:]
plt.figure(figsize=(10,6))
sns.scatterplot(x='long', y='lat', data = bottom_ninety_nine_perc, hue = 'price', edgecolor=None, alpha=0.2, palette='RdYlGn')
plt.show()#we basically recreated the map of King County, Seattle 
sns.boxplot(x='waterfront', y='price', data = df)
plt.show()

#now we drop unnecesary data 

df = df.drop('id', axis = 1)
df['date'] = pd.to_datetime(df['date']) #now we can extract month or year automatically 
df['year'] = df['date'].apply(lambda date: date.year)
df['month'] = df['date'].apply(lambda date: date.month)
plt.figure(figsize=(10,6))
sns.boxplot(x='month', y='price', data = df)
plt.show()
#or we could just do 
df.groupby('month').mean()['price'].plot()
plt.show()
df.groupby('year').mean()['price'].plot()
plt.show()
df=df.drop('date', axis = 1)
df = df.drop('zipcode', axis=1)

#data preprocessing

X = df.drop('price', axis = 1).values
y = df['price'].values
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=101)
print(X_train.shape)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = Sequential()
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(x=X_train, y = y_train, validation_data =(X_test, y_test) ,batch_size=128, epochs = 400)

#model evaluations and predictions
loss_df = pd.DataFrame(model.history.history)
print(loss_df) #now we get the loss on the training set and val_loss - the  lost on the test set in order to see if you're over fitting 
loss_df.plot()
plt.show()
predictions = model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, predictions)))
print(mean_absolute_error(y_test,predictions))

print(explained_variance_score(y_test, predictions))
plt.scatter(y_test, predictions)
plt.plot(y_test, y_test, 'r')
plt.show()






