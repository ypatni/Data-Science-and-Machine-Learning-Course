import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

df3 = pd.read_csv('df3')
mpl.rcParams['patch.force_edgecolor'] = True
sns.set()
print(df3.head())
df3.plot.scatter(x = 'a', y='b', c= 'red', s = 50, figsize=(8,3) )
plt.show()
#2
df3['a'].plot.hist()
plt.show()
#3
plt.style.use('ggplot')
df3['a'].plot.hist(bins= 30, alpha=0.5)
plt.show()
#4
df3[['a','b']].plot.box()
plt.show()
