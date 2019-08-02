import pandas as pd
import numpy as np

#generating a dataframe
dates = pd.date_range('20150101',periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=['a','b','c','d'])
print(df)

#output df's a column,just using column name

#print(df['a'])
#print(df.a)
#using slicing
#print(df[0:3])
# print(df['20150101':'20150104'])

#using label parameter(local),just output the data of index 20150101

#print(df.loc['20150101'])
# just chosing a ,b column, but with all the row
#print(df.loc[:,['','b']])
#chosing with specific row and column
# print(df.loc['20150101',['a']])

#select by postion: iloc,using index to output
# print(df.iloc[3:5,1:3])
# print(df.iloc[[1,3,5],[1,2]])

#select by mix(postion and label):ix
# print(df.ix[:3,['a','b']])

# boolean indexing,we can set a judgement for choosing data
# print(df[df.a>5])

# setting value . using method to access data then change that shit
# df.iloc[2,2] = 111
# commom use , assign the vaule to 0 when df's column bigger than 4
# df[df.a>4] = 0
# df.a[df.a>4] = 0
# df.b[df.a>4] = 0
# print(df)
# add more column
df['F'] = np.nan
df['E'] = pd.Series([1,2,3,4,5,6],index=dates)
print(df)

