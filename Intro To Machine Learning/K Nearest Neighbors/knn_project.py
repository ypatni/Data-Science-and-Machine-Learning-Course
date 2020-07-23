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

scaled_features = scaler.transform(df.drop('TARGET CLASS', axis = 1))

df_feat = pd.DataFrame(scaled_features, columns = df.columns[:-1]) 
print(df_feat.head())
x= df_feat 
y= df['TARGET CLASS']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
error_rate = [] 
for i in range(1,60):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,60), error_rate, color = 'green', linestyle = 'dashed', marker = 'o', markerfacecolor = 'blue', markersize = 10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

#reasonably low error rate at k =39 
knn = KNeighborsClassifier(n_neighbors= 39)
knn.fit(x_train, y_train)
pred = knn.predict(x_test)
print(classification_report(y_test, pred)) 
print(confusion_matrix(y_test, pred)) 