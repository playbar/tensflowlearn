import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
'''
主要是matplotlib的东西
继续学习'''
#Series
# data = pd.Series(np.random.randn(1000),index=np.arange(1000))
# data = data.cumsum()
# data.plot()
# plt.show()

#DataFrame
data = pd.DataFrame(np.random.randn(1000,4),index=np.arange(1000),columns=list("abcd"))
# print(data.head(10))
data = data.cumsum()
data.plot()
plt.show()