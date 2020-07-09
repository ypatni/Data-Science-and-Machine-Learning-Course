import numpy as np 
import pandas as pd 
from sqlalchemy import create_engine
pd.read_csv('example')

#print(pd.read_csv('example'))

df = pd.read_csv('example')
df.to_csv('My_Output', index= False)#index=False to make sure that the column doesn't save the index 

#pandas can only import data with excel files 
print(pd.read_excel('Excel_Sample.xlsx', sheet_name= 'Sheet1'))
df.to_excel('Excel_Sample2.xlsx', sheet_name= 'NewSheet')

#html files
data = pd.read_html('https://www.fdic.gov/Bank/individual/failed/banklist.html')
print(data[0].head())

#sql
engine = create_engine('sqlite:///:memory:')
df.to_sql('my_table', engine)
print(pd.read_sql('my_table', con=engine))