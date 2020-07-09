import numpy as np 
import pandas as pd 

df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})

#finding unique values 
print(df.head())
print(f"\n")
print(df['col2'].unique())
print(df['col2'].nunique()) # for number of unique values 
print(df['col2'].value_counts()) # returns number of times each element is present 

#apply method
print(f"\n")

print(df['col1'].apply(lambda x: x*2))
print(df['col3'].apply(len)) #you can also use built in functions 

print(f"\n")
print(df.drop('col1', axis = 1))
print(df.columns)
print(df.index)

#sorting values 
print(f"\n")
print(df.sort_values('col2'))

