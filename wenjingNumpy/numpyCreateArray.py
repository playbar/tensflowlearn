import numpy as np

# numpy create an array from list

array = np.array([[1,2,3],[2,3,4]])

# then numpy will have some property for the matrix
print(array)
print("number of dim",array.ndim)
print("shape",array.shape)
print("size",array.size)

# generate zeros\ones\empty matrix ,with the input is the shape of the matrix
print(np.zeros((3,4)))

print(np.ones((3,4)))

print(np.empty((3,4)))


# generate array ,with range 10-20 and step 2
a = np.arange(10,20,2)
print(a)

# generate matrix from 0-12 ,with shape 3*4
print(np.arange(12).reshape((3,4)))


# generate linespace from range with step

print(np.linspace(1,10,6).reshape(2,3))


#ndarray
dataset = np.ndarray(shape=(3,3,3),dtype=np.float32)
print("dataset's size",dataset.shape)


