import matplotlib.pyplot as plt
import numpy as np
'''
就是简单的把x y 对应的关系plot出来'''
x = np.linspace(-1,1,50)
y = 2*x +1
plt.plot(x,y)
plt.show()