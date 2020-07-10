import matplotlib.pyplot as plt
import numpy as np 
x = np.linspace(0,5,11)
y = x ** 2

fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])
ax.plot(x,y,color= 'orange', lw=2, alpha= 0.5, ls='--', marker='o', markersize=9, 
        markerfacecolor= "red", markeredgewidth=2, markeredgecolor= 'black') #add hex codes for custom colors, lw = linewidth, ls = linestyle
ax.set_xlim(0,1) #setting lower and upper bounds for an axis
ax.set_ylim(0,2)
plt.show()



