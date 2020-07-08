import numpy as np

arr = np.arange(25)
arr2 = np.random.randint(0,50,10)

#reshape
print(arr.reshape(5,5)) #total size of  array must be unchanged 

#max and min
print(arr2.max())
print(arr2.min())

#index of max and  value 
print(arr2.argmax())
print(arr2.argmin())

print(arr.shape)

#findind datatype of array 
print(arr.dtype)

