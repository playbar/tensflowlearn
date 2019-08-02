import numpy as np

A = np.arange(3,15).reshape((3,4))
print(A)
print(A[2])
# 表示打印第三行的所有数
print(A[2,:])
#表示所有的第2列
print("A[:1]")
print(A[:,1])
# numpy 矩阵中[X,Y] x永远代表第几行 y永远代表第几列， 如果：表示全部的 ，0表示第一个 -1表示最后一个
# 对行迭代
for row in  A:
    print(row)
# 对列迭代
for col in A.T:
    print(col)
# 对每个元素迭代
for item in A.flat:
    print(item)