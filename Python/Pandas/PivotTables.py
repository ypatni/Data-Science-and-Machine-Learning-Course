import numpy as np 
import pandas as pd 

#pivot table 
data = {'A':['foo','foo','foo','bar','bar','bar'],
         'B':['one','one','two','two','one','one'],
           'C':['x','y','x','y','x','y'], 
           'D':[1,3,2,5,4,1]}
df2 = pd.DataFrame(data)
print(df2.pivot_table(values='D', index= ['A', 'B'], columns= ['C']))

