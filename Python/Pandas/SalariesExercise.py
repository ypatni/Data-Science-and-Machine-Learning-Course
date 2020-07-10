import pandas as pd 
import numpy as np 

pd.read_csv('Salaries.csv')

sal = pd.read_csv('Salaries.csv')

print(sal.head())
print(sal.info())
print(f"\n")
print(sal['BasePay'].mean())
print(f"\n")
print(sal['OvertimePay'].max())
print(f"\n")
print(sal[sal [ 'EmployeeName'] == "JOSEPH DRISCOLL"]['JobTitle'])
print(f"\n")
print(sal[sal [ 'EmployeeName'] == "JOSEPH DRISCOLL"]['TotalPayBenefits'])
print(f"\n")
print(sal[sal [ 'TotalPayBenefits'] == sal['TotalPayBenefits'].max()]['EmployeeName'])
print(f"\n")

print(sal[sal [ 'TotalPayBenefits'] == sal['TotalPayBenefits'].min()])
print(f"\n")
print(sal.groupby('Year').mean()['BasePay'])
print(f"\n")
print(sal['JobTitle'].nunique())
print(f"\n")
print(sal['JobTitle'].value_counts().head(5))
print(f"\n")
# How many Job Titles were represented by only one person in 2013? (e.g. Job Titles with only one occurence in 2013?)
print(sum(sal[sal['Year'] == 2013]['JobTitle'].value_counts() ==1 )) 
#How many people have the word Chief in their job title? (This is pretty tricky)
print(f"\n")
#print(sum(sal['JobTitle'].str.lower().str.contains('chief'))) - one solution 
def chief_name(title):
    if 'chief' in title.lower():

        return True

    else:

        return False  

print(sum(sal['JobTitle'].apply(lambda x: chief_name(x))))

#Is there a correlation between length of the Job Title string and Salary?
sal['title_len'] = sal['JobTitle'].apply(len)
print(sal[['TotalPayBenefits', 'title_len']].corr())


