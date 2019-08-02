import pandas as pd
import numpy as np

# 会按照0 value ，1  value ，2 value 这样的形式把数据创建出来
# s = pd.Series([1,3,6,np.nan,44,1])
# print (s)

dates = pd.date_range('20160101',periods=6)
df = pd.DataFrame(np.random.randn(6,4),index =dates,columns=['a','b','c','d'])

# 真他妈强大。
#                    a         b         c         d
# 2016-01-01 -2.373797 -1.510231  0.796347 -1.086612
# 2016-01-02  0.514725  0.062522  0.960518 -2.185785
# 2016-01-03  0.730802  0.504954 -0.572622 -0.759270
# 2016-01-04 -1.962342  0.626760 -0.703240  0.488508
# 2016-01-05 -1.181061 -1.212350 -0.072592  0.190871
# 2016-01-06 -1.003676 -0.214570  0.065969 -0.290509
print(df)

# 可以指定 行 列 的index，便于显示
index2 = pd.Categorical(["shabi","nima","dineng","train"])
df2 = pd.DataFrame({'A':1.,'B':pd.Timestamp('20130202'),'C':pd.Series(1,index=list(range(4)),dtype='float32'),
                    'D':np.array([3]*4,dtype='int32'),'E':pd.Categorical(["test","train","test","train"]),
                    'F':'foo'},index=index2)
print(df2)

#创建了dataFrame， 就可以有很多的内部方法。
print(df2.columns)
print(df2.values)
#然后可以根据进行各种操作，比如sort,以column 表示axis =1 ，
print(df2.sort_index(axis=1,ascending=False))
#以值做排序
print(df2.sort_values(by='E'))