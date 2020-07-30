import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

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
