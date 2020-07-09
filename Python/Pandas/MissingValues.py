import numpy as np 
import pandas as pd 

d = {'A': [1,2,np.nan], 'B':[5, np.nan, np.nan] ,'C': [3, 6, 9]}
df = pd.DataFrame(d)
print(df)
print(f"\n")
print(df.dropna()) # dropping all rows with null values 
# use axis=1 argument to dropp an columns with null values 
#use thresh argument to set a threshold of non  NaN values 
print(f"\n")
print(df.dropna(thresh=2)) # kept all rows with at least 2 non NaN values 


#filling NaN values 
#fill with random string
print(f"\n")
print(df.fillna(value = "HELLO"))
#fill with mean of column 
print(f"\n")
print(df['A'].fillna(value = df['A'].mean()))