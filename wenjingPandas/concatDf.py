import numpy as np
import pandas as pd
# concatenating 简单粗暴的方法直接把df相连接
#concatenating
# df1 = pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'])
# df2 = pd.DataFrame(np.ones((3,4))*1,columns=['a','b','c','d'])
# df3 = pd.DataFrame(np.ones((3,4))*2,columns=['a','b','c','d'])
# # print(df1)
# # print(df2)
# # print(df3)
# res = pd.concat([df1,df2,df3],axis=0,ignore_index=True)
# print(res)

# join ,['inner','outer']
df1 = pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1,columns=['a','b','c','d'])
print(df1)
print(df2)
# join 表示连表，默认用outer, inner 会自动裁剪掉不重复的，只保留重复的
# res = pd.concat([df1,df2],axis=0,join = 'inner',ignore_index=True)
# print(res)
# 以某个df的index作为合并以后的index
res = pd.concat([df1,df2],axis=1,join_axes=[df1.index])
print(res)


#append
# res = df1.append(df2,ignore_index=True)
#合并多个df添加
# res = df1.append([df2,df2],ignore_index=True)
# print(res)

#append single line
s1 = pd.Series([1,2,3,4],index=['a','b','c','d'])
res = df1.append(s1,ignore_index=True)
print(res)