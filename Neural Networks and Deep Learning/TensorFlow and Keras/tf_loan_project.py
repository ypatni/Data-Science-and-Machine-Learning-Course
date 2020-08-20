import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

mpl.rcParams['patch.force_edgecolor'] = True
sns.set()
sns.set_style('whitegrid')
data_info = pd.read_csv('DATA/lending_club_info.csv', index_col='LoanStatNew')

def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])
print(feat_info('mort_acc'))

df = pd.read_csv('DATA/lending_club_loan_two.csv')
print(df.describe().transpose())
sns.countplot(x='loan_status', data= df)
plt.show()
sns.distplot(df['loan_amnt'], bins = 30, kde=False)
plt.show()
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap= 'viridis')
plt.show()
feat_info('installment')
sns.scatterplot(x='installment', y = 'loan_amnt', data=df, alpha =0.6, palette='RdYlGn')
plt.show()
print(df.groupby('loan_status')['loan_amnt'].describe())
#grades and subgrades 
print(df['sub_grade'].unique())
sns.countplot(x='grade',data=df,hue='loan_status')
plt.show()
plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order = subgrade_order,palette='coolwarm' )
plt.show()
f_and_g = df[(df['grade']=='G') | (df['grade']=='F')]

plt.figure(figsize=(12,4))
subgrade_order = sorted(f_and_g['sub_grade'].unique())
sns.countplot(x='sub_grade',data=f_and_g,order = subgrade_order,hue='loan_status')
plt.show()
df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})
print(df[['loan_repaid','loan_status']])
df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')
plt.show()
print(df.isnull().sum())
#we want to see the missing numbers in terms of percentage of dataframe values 
print(df.isnull().sum()/len(df))
feat_info('emp_title')
print(df['emp_title'].nunique()) #theres just way too many titles to make dummy variables 
#might as well just remove it 
df = df.drop('emp_title', axis = 1)
print(sorted(df['emp_length'].dropna().unique()))
emp_length_order = [ '< 1 year','1 year','2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years', '10+ years']
sns.countplot(x='emp_length', data=df, order=emp_length_order, he = 'loan_status')
plt.show()
#we need to find percent of charge offs per category to figure  out is there is a strong relationship with emp_length
emp_co = df[df['loan_status ']=='Charged Off'].groupby('emp_length').count()['loan_status']
emp_fp = df[df['loan_status ']=='Fully Paid'].groupby('emp_length').count()['loan_status']

emp_len = emp_co/emp_fp
print(emp_len)
emp_len.plot(kind='bar')
plt.show() #the bars are pretty similar, showing no reason to keep this feature 
df = df.drop('emp_length', axis = 1)
df = df.drop('title', axis = 1)
#trying to fill more missing data 
print(df['mort_acc'].value_counts()) #theres a lot of missing values from this account so we can't drop the rows 
#checking for correlation 
print(df.corr()['mort_acc'].sort())
#total_acc has pretty good positive correlation 
#group df by total_acc and calculate mean value for he mort_acc per total_acc entry 
total_acc_means = df.groupby('total_acc').mean()['mor_acc']
print(total_acc_means)
def fill_mort_acc(total_acc, mort_acc):
    if np.isnan(mort_acc):
        return total_acc_means[total_acc]
    else:
        return mort_acc
df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis = 1) 

df = df.dropna()
print(df.isnull().sum())
#No more missing data 










