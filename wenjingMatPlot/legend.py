import matplotlib.pyplot as plt
import numpy as np
'''
legend 是坐标的小标图的意思'''
x = np.linspace(-3,3,50)
y = 2*x +1
y2 = x**2

plt.figure(num=3,figsize=(8,5))


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

# plt.plot(x,y2,label='up')
# plt.plot(x,y,color='red',linewidth = 1.0,linestyle='--',label='down')
l1=plt.plot(x,y2,label='up')
l2=plt.plot(x,y,color='red',linewidth = 1.0,linestyle='--',label='down')
#默认展示 直接显示label的值作为legend ，也可以自定义参数
#plt.legend()
# best means best location other parameter , upper right...balabala
# 这样会忽视原来的label， 用新的labels ，然后handles 需要之前plt plot出来的返回值
# 也可以只显示 L1  只放入L1即可
plt.legend(handles=[l1,l2,],labels=['aaa','bbb'],loc='best')
plt.show()