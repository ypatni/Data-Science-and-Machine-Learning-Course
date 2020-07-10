import matplotlib.pyplot as plt
import numpy as np 

#using OO

x = np.linspace(0,5,11)
y = x ** 2

fig, axes = plt.subplots(nrows=1, ncols=2) #you can specify number of rows and columns! 
#axes is basically a list of matplotlib axes objects! so you can iterate through it and index it 
for i in axes: 
    i.plot(x,y) # makes the same plot depending on number of rows and columns over and over
plt.tight_layout()
plt.show()



