import matplotlib.pyplot as plt
import numpy as np 

#using OO and adding legends


x = np.linspace(0,5,11)
y = x ** 2

fig = plt.figure()
ax = fig.add_axes([0.1,0.1,1,1])
ax.plot(x,y, label = "X Squared")
ax.plot(x,x**3, label= "X Cubed")
ax.legend(loc=(0.05,0.75))
plt.tight_layout()
plt.show()

#saving figures to a file

fig.savefig('mypicture.png')
