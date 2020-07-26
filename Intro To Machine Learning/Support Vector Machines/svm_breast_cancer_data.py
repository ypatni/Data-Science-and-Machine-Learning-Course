import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
mpl.rcParams['patch.force_edgecolor'] = True
sns.set()
sns.set_style('whitegrid')
cancer = load_breast_cancer()
#print(cancer['DESCR'])
#we are trying to predict if the cancer is malignant of beningn 
df_feat = pd.DataFrame(cancer['data'], columns = cancer['feature_names'])
print(df_feat.head())