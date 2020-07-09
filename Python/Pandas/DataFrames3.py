import numpy as np 
import pandas as pd 
from numpy.random import randn 

outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]
hier_index = list(zip(outside, inside)) #making into list of tuple pairs 
hier_index = pd.MultiIndex.from_tuples(hier_index) # takings a list and creates multi index from it 

df = pd.DataFrame(randn(6,2), hier_index, ['A','B'])
print(df)

# calling data from a multi level hierarchy 
print(f"\n")
print(df.loc['G1'].loc[1]) # passed as a series, go from outside to index 
df.index.names = ['Groups', 'Num']
print(df)

