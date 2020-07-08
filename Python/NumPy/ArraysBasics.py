import numpy 

print(numpy.arange(1,10))
print(numpy.arange(0,11,2)) # with step size 
print(numpy.ones((2,3)))
print(numpy.linspace(1,5,10)) #evenly spaced points between first two parameters
print(numpy.eye(4)) #2d square matrix with diagonal of ones 

#creating random arrays 
print(numpy.random.rand(6)) #values between zero and one
print(numpy.random.rand(5,5)) # 2D

print(numpy.random.randn(3))

print(numpy.random.randint(1,20, 5)) # lowest inclusive highest exclusive



