import matplotlib.pyplot as plt
import numpy as np 
x = np.linspace(0,5,11)
y = x ** 2

#functional method  - super basic 
plt.plot(x,y, 'r-') 
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Title')

plt.subplot(1,2,1) #num rows, num columns, plot number 
plt.plot(x,y,'r')
plt.subplot(1,2,2)
plt.plot(y,x,'b')
plt.show()

