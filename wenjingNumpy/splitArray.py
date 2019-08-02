import  numpy as  np
A = np.arange(12).reshape((3,4))

print(A)

#按照列，分割成2块
print(np.split(A,2,axis=1))

# 按照列，不等分为3块，这个不会报错
print(np.array_split(A,3,axis=1))

# 类似 hstack Vstack， hsplit，vsplit表示横向，或者纵向
print(np.vsplit())
