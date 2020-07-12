import pandas as pd 
import numpy as np 
import chart_studio.plotly as py 
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


data = dict(
    type='choropleth', #type of geographical plot 
    locations= ['AZ', 'NY', 'CA'], #list of state abbreviation codes
    locationmode= 'USA-states', #can go down to county levels
    colorscale='reds',
    text=['text1', 'text2', 'text3'], #list of text that lines up for each of the locations
    z=[1.0,2.0,3.0], # value that you want to represent as a color to indicate levels 
    colorbar= {'title': 'Colorbar goes here'} 
)
layout = dict(geo = {'scope':'usa'})

fig = go.Figure(
    data= [data], 
    layout = layout
)
fig.show()

