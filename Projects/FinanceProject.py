from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import chart_studio.plotly as py 
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, plot, iplot

start = datetime.datetime(2006, 1, 1)
end = datetime.datetime(2016,1,1)
mpl.rcParams['patch.force_edgecolor'] = True

sns.set()


# Bank of America
BAC = data.DataReader("BAC", 'stooq', start, end)
# CitiGroup
C = data.DataReader("C", 'stooq', start, end)
# Goldman Sachs
GS = data.DataReader("GS", 'stooq', start, end)
# JPMorgan Chase
JPM = data.DataReader("JPM", 'stooq', start, end)
# Morgan Stanley
MS = data.DataReader("MS", 'stooq', start, end)
# Wells Fargo
WFC = data.DataReader("WFC", 'stooq', start, end)

tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']

bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC], axis = 1, keys = tickers)
bank_stocks.columns.names = ['Bank Ticker','Stock Info']
#print(bank_stocks.head())

print(bank_stocks.xs(key='Close',axis=1,level='Stock Info').max())
returns = pd.DataFrame()
#making a dataframe with the returns from each stock 
for i in tickers: 
    returns[i+' Return'] = bank_stocks[i]['Close'].pct_change()
print(returns.head())

#sns.pairplot(returns[1:])

#plt.show()
print(returns.idxmin())
print(f"\n")
print(returns.std()) #citibank riskiest stock 
print(f"\n")
print(returns['2015-01-01':'2015-12-31'].std()) #BofA or Morgan Stanley
sns.distplot(returns['2015-01-01':'2015-12-31']['MS Return'],color='green',bins=100)
plt.show()

bank_stocks.xs(key='Close',axis=1,level='Stock Info').iplot()
plt.show()

