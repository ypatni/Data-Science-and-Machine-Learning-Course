#exercises 
import numpy 

print(numpy.zeros(10))
print(numpy.arange(10,51, 2))

print(numpy.arange(0,9).reshape(3,3))
print(numpy.random.randn(1))

print(numpy.linspace(0,1,100).reshape(10,10))


arr_2d = numpy.arange(1,26).reshape(5,5)
print(arr_2d)
print(arr_2d[2:,1:])
print(arr_2d[0:3,1:2])
print(numpy.sum(arr_2d))
print(numpy.std(arr_2d))
print(numpy.sum(arr_2d, axis=0))