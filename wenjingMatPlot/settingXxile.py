import matplotlib.pyplot as plt
import numpy as np
'''
基本的X Y 轴操作
包括设定范围，改变量名 matching到字符串
移动X Y 轴'''
x = np.linspace(-3,3,50)
y = 2*x +1
y2 = x**2

plt.figure(num=3,figsize=(8,5))
plt.plot(x,y2)
plt.plot(x,y,color='red',linewidth = 1.0,linestyle='--')

# setting  axle range
plt.xlim((-1,2))
plt.ylim((-2,3))
plt.xlabel("i am x")
plt.ylabel("i am y")

# 指定脚标，从-1，到2 5个区间，也就是5个脚标，
new_ticks = np.linspace(-1,2,5)
print(new_ticks)
plt.xticks(new_ticks)
# 可以根据值 设置lable的字符
plt.yticks([-2,-1,0,1,2],
           ['re bad','bad','normal','good','really good'])

# gca = get current axis
ax = plt.gca()
# spines means the frame of your axis
# let right side and top side color none
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
##设置 X Y 分别为 ax spine里的bottom 和left
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
# 把bottom 位置设在 y data 为0的地方,把left 的位置设在x data 0的地方，这样就变成了坐标原点 为0 0
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))
plt.show()