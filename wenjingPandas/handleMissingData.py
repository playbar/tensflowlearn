import  pandas as pd
import  numpy as np

dates = pd.date_range('20130101',periods=6)
df = pd.DataFrame(np.arange(24).reshape(6,4),index=dates,columns=['a','b','c','d'])
df.iloc[0,1] = np.nan
df.iloc[2,1] = np.nan
# axis means row or colomn ,how means how to drop , any means anyone all means everyone
# dropna means drop the ma value
print(df.dropna(axis=0,how='all'))
#fill the data with X to the posion of 0
# print(df.fillna(value=0))
#print out all the data if there is a null value
# print(df.isnull())
# check if any number is null in the df
print(np.any(df.isnull())==True)
