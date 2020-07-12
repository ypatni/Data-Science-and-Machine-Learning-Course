import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import seaborn as sns


df1 = pd.read_csv('df1', index_col = 0)
#print(df1.head())
df2 = pd.read_csv('df2')
#print(df2.head())
mpl.rcParams['patch.force_edgecolor'] = True

sns.set()


df1['A'].hist(bins=30)

df1['A'].plot(kind='hist')
df1['A'].plot.hist()

plt.show()
#area plots 
df2.plot.area(alpha= 0.6)
plt.show()
#bar graphs 
df2.plot.bar(stacked=False)
plt.show()
#line plots 
df1.plot.line( y='B', figsize=(12,3), lw = 3) 
plt.show()
#scatter plots 
df1.plot.scatter(x = 'A', y='B', c='C', cmap= 'coolwarm') #here the color 'c' is based off of the column 'C' making a 3D plot   
plt.show()
df1.plot.scatter(x = 'A', y='B', c='C', s=df1['C']*100) #here you use the size of the points to show their correlation to C 
plt.show()
#box plots 
df1.plot.box()
plt.show()
#hexagonal bin plots 
df = pd.DataFrame(np.random.randn(1000,2), columns = ['a', 'b'])
df.plot.hexbin(x='a', y='b', gridsize=25, cmap='coolwarm')
plt.show()
#density estimation plot 
df2.plot.kde()
plt.show()



