import numpy as np

a = np.array([10,20,30,40])
b = np.arange(4)

# basic minus and sum
# c = a - b
# c = a + b
# pow
# c = a**2
# get sin or cos or tan ..
# c = 10*np.sin(a)
# 如果对一个array 进行比较操作，他会对每一个数组里的数进行一次比较操作
# print(b<5)
# print(a)
# print(b)
# print(c)

# 逐个相乘
c = a*b
# 矩阵相乘
c = np.dot(a,b)

# np 内部的方法 random.random 是随机生成shape的矩阵，值0-1不等
a = np.random.random((2,4))
# np.sum np.max np.min mean average,cumsum(累加) diff（每2个数之间的差），transpose（转置）
# nonzero(非零的数，输出2个array，表示第n行第j列非0) sort（排序） clip（参数，把矩阵超出范围的剪掉）
# np 中调用参数有一个特别的 叫axis 如果axis = 1 表示每一行之中， axis = 0 表示每一个列之中

# 返回搜索最小/大值的索引
A = np.arange(2,14).reshape((3,4))
np.argmin(A)
np.argmax(A)




