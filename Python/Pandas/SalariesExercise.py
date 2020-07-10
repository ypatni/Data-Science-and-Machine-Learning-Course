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



