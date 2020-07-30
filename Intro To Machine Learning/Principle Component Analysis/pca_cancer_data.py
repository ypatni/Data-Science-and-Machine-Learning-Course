import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
mpl.rcParams['patch.force_edgecolor'] = True
sns.set()
sns.set_style('whitegrid')
cancer = load_breast_cancer()
print(cancer.keys())
# print(cancer['DESCR'])
#we're trying to find which components are the most important ones that explain the most variance of the dataset 
df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
print(df.head())

scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)

#PCA 
pca = PCA(n_components = 2)
pca.fit(scaled_data)
for i in range(0, 2):
    for j in range(0, pca.components_[i].size):
        pca.components_[i][j] = -1 * pca.components_[i][j]
x_pca = pca.transform(scaled_data)
plt.figure(figsize = (8,6))
plt.scatter(x_pca[:,0],x_pca[:,1], c = cancer['target'], cmap= 'rainbow' )

plt.xlabel('First Principle Component')
plt.ylabel('Second Principle Component')
plt.show()