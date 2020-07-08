import numpy as np

arr = np.arange(0,11)
bool_arr = arr>5

print(arr[bool_arr])

arr_2d = np.arange(1,51).reshape(5,10)
print(arr_2d)