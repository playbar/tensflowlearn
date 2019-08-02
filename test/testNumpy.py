#-*- coding =utf-8 -*-
import numpy as np


# x_data =np.linspace(-1,1,300)
#
# print (x_data)
# 生成 一个数组，-1,1 300个

# x_data =np.linspace(-1,1,300)[:,np.newaxis]
# 通过这个操作 把每一个生成的点，变为一个单独的数组
# print (x_data)


pred = [[2],[3],[5]]
print(pred)
result = np.reshape(pred,[-1])
print (result)