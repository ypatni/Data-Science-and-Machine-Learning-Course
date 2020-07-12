import pandas as pd 
import numpy as np 
import chart_studio.plotly as py 
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, plot, iplot

df = pd.read_csv('2012_Election_Data')
#print(df.head())
data = dict(
    type = 'choropleth',
    colorscale= 'reds',
    locations= df['State Abv'],
    locationmode= 'USA-states',
    z = df['Voting-Age Population (VAP)'], 
    text = df['State'],
    marker= dict(line = dict(color = 'rgb(12, 12, 12)', width = 1)),
    colorbar = {'title':'VAP' }
)
layout = dict(title = "Voting Age Population By State", geo = dict(scope = 'usa', showlakes= True, lakecolor= 'rgb(85, 173, 240)'))
fig = go.Figure(
    data = [data], 
    layout = layout
)
fig.show()