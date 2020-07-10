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

fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].plot(x,y)
axes[1].plot(y,x) # this just shows how each subplot is basically a part of an array that you can individually access and set making it so much more efficient 
axes[0].set_title('First Plot')
axes[1].set_title('Second Plot')
plt.tight_layout() # to stop axes overlapping
plt.show()
#axes.plot(x,y)

