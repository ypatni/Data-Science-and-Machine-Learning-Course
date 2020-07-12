import pandas as pd 
import numpy as np 
import chart_studio.plotly as py 
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, plot, iplot

df = pd.read_csv('2011_US_Agri_Exports')
#print(df.head())

data = dict(
    type = 'choropleth',
    colorscale= 'reds',
    locations= df['code'],
    locationmode= 'USA-states',
    z = df['total exports'], 
    text = df['text'],
    marker= dict(line = dict(color = 'rgb(255, 255, 255)', width = 2)),
    colorbar = {'title':'Millions USD' }
)
layout = dict(title= '2011 US Agriculture Exports By States', geo= dict(scope = 'usa', showlakes= True, lakecolor= 'rgb(85, 173, 240)'))

fig = go.Figure(
    data= [data],
    layout= layout
)
fig.show()