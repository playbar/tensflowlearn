import  numpy as np

A = np.array([1,1,1])
b = np.array([2,2,2])

# vstack need a tuble parameter
#v means vertical stack
c = np.vstack((A,b))
print(c)
#[1,1,1,2,2,2] 序列和序列合并，还是序列
d = np.hstack((A,b))
print(d)
#如何把[1,1,1] 变成[1][1][1] ,不能用A.T 没用
print(A[np.newaxis,:])
#表示在A的行上加了一个维度 [[1,1,1]],也就是这一行我们放一起，加一个[]
print(A[:,np.newaxis])
#[[1][1][1]]
#表示在列上加了一个维度 ， 也就是每一列，我们都加一个[]
# 还有一个合并 concatenate（()) 他可以指定方向进行合并 用axis 1 表示横向 ， 0表示竖向



