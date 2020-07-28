import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
mpl.rcParams['patch.force_edgecolor'] = True
sns.set()
sns.set_style('whitegrid')

#we need to specify the number of clusters to be used

data = make_blobs(n_samples = 200, n_features = 2, centers = 4, cluster_std = 1.8, random_state = 101)
print(data[0].shape) #gives you a tuple with numpy arrays here it has 200 samples with 2 features

plt.scatter(data[0][:,0], data[0][:,1], c = data[1], cmap = 'rainbow') #all rows in first column vs all rows in second column and then we use the centers 
plt.show()

