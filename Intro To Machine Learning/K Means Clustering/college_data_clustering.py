import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
mpl.rcParams['patch.force_edgecolor'] = True
sns.set()
sns.set_style('whitegrid')

df = pd.read_csv('College_Data', index_col = 0)
print(df.head())
sns.lmplot('Room.Board', 'Grad.Rate', data = df, hue = 'Private', palette = 'coolwarm', height = 7)
plt.show()
sns.lmplot('Outstate', 'F.Undergrad', data = df, hue = 'Private', height = 7, fit_reg=False)
plt.show()
sns.set_style('darkgrid')
g = sns.FacetGrid(df, height = 7,  hue="Private")
g.map(plt.hist, 'Outstate', bins = 20, alpha = 0.6)
plt.show()
sns.set_style('darkgrid')
g = sns.FacetGrid(df, height = 7,  hue="Private")
g.map(plt.hist, 'Grad.Rate', bins = 20, alpha = 0.6) 
# ^ stacked histogram
plt.show()
print(df[df['Grad.Rate']>100]) #we found one schools graduation rate to be higher than 100 for some reason 
#resetting to 100 
df['Grad.Rate']['Cazenovia College'] = 100
