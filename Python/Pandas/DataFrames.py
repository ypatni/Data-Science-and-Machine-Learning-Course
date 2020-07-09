import numpy as np 
import pandas as pd 
from numpy.random import randn 

a = np.random.seed(101)
df = pd.DataFrame(randn(5,4), ['A', 'B', 'C', 'D', 'E'], ['W', 'X', 'Y', 'Z'])
print(df)

#indexing
#finding a single column 
print(f"\n")
print(df['W'])
# print(type(df['W'])) - series type 

#using lists to get multiple colums 
print(df[['X','Y']])

df['new'] = df['X'] + df['Y']
print(df)

#dropping a column 
print(f"\n")
df.drop('new', axis=1, inplace = True) #inplace=True permanently removes the column/row mentioned\
print(df)

#dropping a row
print(f"\n")
print(df.drop('E'))

#ROWS 
print(f"\n")
print(df.loc['C'])
print(df.iloc[2]) #numerical based index 
print(f"\n")
print(df.loc['B', 'Y'])

print(f"\n")
print(df.loc[['A', 'B'], ['W', 'Y']])