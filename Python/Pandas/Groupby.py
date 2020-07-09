#groupby allows you to group wrows based on a column and then perform an aggregate function on them 

import numpy as np 
import pandas as pd 
data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],'Sales':[200,120,340,124,243,350]}
df = pd.DataFrame(data)
print(df)
by_comp  = df.groupby('Company')
print(f"\n")
print(by_comp.mean()) #ignores non numeric column 
print(by_comp.std()) #perfoming basic operations is easier now 
print(f"\n")

print(df.groupby('Company').sum().loc['FB'])
print(df.groupby('Company').count()) 
print(df.groupby('Company').max()) # also returns the string of the person with the name closest to the end of the alphabet
print(f"\n")
print(df.groupby('Company').describe().transpose()['FB']) #gives a TON of information
