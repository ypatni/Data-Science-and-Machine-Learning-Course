import pandas as pd 
import numpy as np 
import chart_studio.plotly as py 
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, plot, iplot

df = pd.read_csv('2014_World_Power_Consumption')
#print(df.head())
data = dict(
    type = 'choropleth', 
    locations = df['Country'],
    z = df['Power Consumption KWH'],
    locationmode = "country names",
    text = df['Country'],
    marker= dict(line = dict(color = 'rgb(12, 12, 12)', width = )),
    colorbar = {'title':'Power Consumption in KWH' },
)
layout = dict(title ='2014 World Power Consumption', height = 700, width = 1300, geo = dict(showframe=False, projection= {'type': 'miller'}) )

fig = go.Figure(
    data = [data],
    layout = layout
)
fig.show()