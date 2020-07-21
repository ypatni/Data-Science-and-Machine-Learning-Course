import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

mpl.rcParams['patch.force_edgecolor'] = True
sns.set()
sns.set_style('whitegrid')

df = pd.read_csv('Classified Data', index_col = 0)
print(df.head())

#when using k nearest neighbors, we need to standardize everything to the same scale

scale = StandardScaler()
scale.fit(df.drop('TARGET CLASS', axis = 1))

scaled_features = scale.transform(df.drop('TARGET CLASS', axis = 1))

df_feat = pd.DataFrame(scaled_features, columns = df.columns[:-1]) 
print(df_feat.head())

x= df_feat 
y= df['TARGET CLASS']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(x_train, y_train)
pred = knn.predict(x_test)
print(classification_report(y_test, pred)) 
print(confusion_matrix(y_test, pred)) 
#so our model was pretty good with k =1, but if we want greater accuracy, we need to figure out which k value is the most 
# appropriate

