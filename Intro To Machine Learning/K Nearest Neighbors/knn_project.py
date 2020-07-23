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

df = pd.read_csv('KNN_Project_Data')
print(df.head())
#sns.pairplot(df, hue = 'TARGET CLASS', palette = 'bwr')
#plt.show() - very laggy 

scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis = 1))

scaled_feat = scaler.transform(df.drop('TARGET CLASS', axis = 1))
