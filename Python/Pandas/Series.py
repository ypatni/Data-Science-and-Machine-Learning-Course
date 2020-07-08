import numpy as np 
import pandas as pd 

labels = ['a', 'b', 'c']
data = [10, 20 , 30 ]
arr = np.array(data)
d = {'a': 10, 'b': 20, 'c': 30}

print(pd.Series(data = arr))
print(f"\n")
print(pd.Series(data = arr, index = labels))
# in this case you could also do this: print(pd.Series(arr, index = labels))

ser1 = pd.Series([1,2,3,4],["USA", "Japan", "India", "Singapore"])
ser2 = pd.Series([1,2,5,4],["USA", "Japan", "France", "Singapore"])
print(f"\n")
print(ser1+ser2)