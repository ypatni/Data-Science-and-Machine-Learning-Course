import pandas as pd 
import numpy as np 

ecom = pd.read_csv('Ecommerce Purchases')
print(ecom)
#print(ecom.head())

print(ecom.info())
print(f"\n")
print(ecom['Purchase Price'].mean())
print(f"\n")
print(ecom['Purchase Price'].max())
print(f"\n")
print(ecom['Purchase Price'].min())
print(f"\n")
print(ecom[ecom['Language']=='en']['Language'].count())
print(f"\n")
print(ecom[ecom['Job']=='Lawyer']['Job'].count())
print(f"\n")
print(ecom['AM or PM'].value_counts())
print(f"\n")
print(ecom['Job'].value_counts().head())
print(f"\n")
print(ecom[ecom['Lot'] == '90 WT']['Purchase Price'])
print(f"\n")
#two conditions
print(ecom[(ecom['CC Provider'] == 'American Express') & (ecom['Purchase Price'] > 95)]['CC Provider'].count())
print(f"\n")
#Hard: How many people have a credit card that expires in 2025?
print(sum(ecom['CC Exp Date'].apply(lambda x: x[3:]) == '25'))
#Hard: What are the top 5 most popular email providers/hosts (e.g. gmail.com, yahoo.com, etc...)
print(f"\n")
print(ecom['Email'].apply(lambda x : x.split('@')[1]).value_counts().head(5))