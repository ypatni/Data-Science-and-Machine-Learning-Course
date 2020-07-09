import numpy as np 
import pandas as pd 
from numpy.random import randn 

np.random.seed(101)
df = pd.DataFrame(randn(5,4), ['A', 'B', 'C', 'D', 'E'], ['W', 'X', 'Y', 'Z'])

# print(df>0) - prints out data frames with true for values >0 and false for values less 

booldf = df>0
print(df[booldf])

print(f"\n{df['W']>0}")
print(df[df['W']>0]) # prints out all values in table excpet the row where W<0 which is row C 
print(f"\n")
print(df[df['W']>0][['Y', 'X']]) #use list notation for more than one value 

#using multiple conditions
print(f"\n")
print(df[(df['W']>0) & (df['Y']>0)])

#reset index
print(f"\n") 
print(df.reset_index())
print(f"\n") 
new_ind = "CA NY OR WA PA".split()
df['States'] = new_ind
print(df)

print(df.set_index('States', inplace=False)) 