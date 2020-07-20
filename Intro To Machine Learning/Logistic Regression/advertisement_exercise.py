import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
mpl.rcParams['patch.force_edgecolor'] = True
sns.set()
ad_data = pd.read_csv('advertising.csv')
print(ad_data.head())
print(ad_data.describe())
sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
plt.show()
#Area Income vs Age
sns.jointplot(x = 'Age', y='Area Income', data = ad_data)
plt.show()
#time spent on site vs age 
sns.jointplot(x = 'Age', y='Daily Time Spent on Site', data = ad_data, color = 'green', kind = 'kde')
plt.show()
sns.pairplot(ad_data,hue='Clicked on Ad',palette='bwr')
plt.show()

