import matplotlib.pyplot as plt
import numpy as np 
x = np.linspace(0,5,11)
y = x ** 2
#using object oriented method allows for more control 

fig = plt.figure() #creating a figure object - basically a blank canvas 
axes = fig.add_axes([0.1,0.1,0.8,0.8]) #adding a set of axes using a list (left, bottom, width, height)
axes.plot(x,y) 
axes.set_xlabel('X Label')
axes.set_ylabel('Y Label')
axes.set_title('Title')
plt.show()

fig = plt.figure()
axis1 = fig.add_axes([0.1,0.1,0.8,0.8])
axis2 = fig.add_axes([0.2,0.5,0.4,0.3])
axis1.plot(x,y)
axis2.plot(y,x)
axis1.set_title('Big Plot')
axis2.set_title('Small Plot')
plt.show()