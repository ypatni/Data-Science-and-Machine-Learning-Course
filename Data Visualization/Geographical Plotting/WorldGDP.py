import pandas as pd 
import numpy as np 
import chart_studio.plotly as py 
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, plot, iplot

df = pd.read_csv('2014_World_GDP')
#print(df.head())

data = dict(
    type = 'choropleth', 
    locations= df['CODE'], 
    z = df['GDP (BILLIONS)'],
    text= df["COUNTRY"], 
    marker= dict(line = dict(color = 'rgb(12, 12, 12)', width = 1)),
    colorbar = {'title':'GDP in Billions USD' }
)
layout = dict(title = '2014 Global GDP',height = 700, width = 1300, geo = dict(showframe=False, projection= {'type': 'miller'}))

fig = go.Figure(
    data = [data], 
    layout= layout
)
fig.show()