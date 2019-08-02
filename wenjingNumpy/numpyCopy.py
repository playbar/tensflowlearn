import numpy as np

a = np.arange(12)
b = a
print (b is a)

c = a.copy()
# 如果只是赋值 要用copy
print(c is a)

