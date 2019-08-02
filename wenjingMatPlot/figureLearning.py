import matplotlib.pyplot as plt
import numpy as np
'''
每次调用figure，就相当生成一张新的图'''
x = np.linspace(-3,3,50)
y = 2*x +1
y2 = x**2

plt.figure()
plt.plot(x,y)
# figure size 就是 长宽比例
plt.figure(num=3,figsize=(8,5))
plt.plot(x,y2)
# 在同一个figure里 可以把多次plot 这样就可以把数据花在一个图中
plt.plot(x,y,color='red',linewidth = 1.0,linestyle='--')
plt.show()