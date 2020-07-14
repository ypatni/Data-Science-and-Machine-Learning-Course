import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import chart_studio.plotly as py 
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, plot, iplot

df = pd.read_csv('911.csv')
mpl.rcParams['patch.force_edgecolor'] = True

sns.set()

print(df.info())

#top 5 zipcodes and townships 
print(df['zip'].value_counts().head(5))
print(df['twp'].value_counts().head(5))
#unique titles 
print(f"\n")
print(df['title'].nunique())
#adding new reason column 
df['Reason'] = df['title'].apply(lambda x: x.split(':')[0])
print(df['Reason'].value_counts())

#creating reason plots 
df['Reason'].value_counts().plot.bar(stacked=False)
plt.show()

print(type(df['timeStamp'].iloc[0]))

df['timeStamp'] = pd.to_datetime(df['timeStamp'])


