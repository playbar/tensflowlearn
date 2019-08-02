import pandas as pd
import numpy as np

'''
merge 比 concat 高级很多，可以基于Index h合并  功能强大
'''
# DataFrame 可以通过 dict来直接生成
left = pd.DataFrame({'key':['a','b','c','d'],
                     'key2':['a','b','c','d'],
                     'A':['1','2','3','4'],
                     'B':['2','3','4','5']})
right = pd.DataFrame({'key':['a','b','c','d'],
                     'key2':['a','b','c','d'],
                     'c':['1','2','3','4'],
                     'd':['2','3','4','5']})
# merging by single key
res = pd.merge(left,right,on='key')
print(res)
# merging on  key list,默认的合并方法是inner，还有left,right ,outer 类似sql的连接
#一个额外的参数 indicator， 显示df是如何被合并的
#一个额外的参数 index ，以df的index来和并
res = pd.merge(left,right,on=['key','key2'],how='inner')
print(res)
#处理overlaping问题的参数 suffixes，可以把overlapping的名字更改，
