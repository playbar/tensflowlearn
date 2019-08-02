# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


'''
周莫烦的第一个TensorFlow demo 1初始化data 2 初始化TensorFlow 结构 2.1 随机生成weight  bias 2.2 lossfunction的选取
2.3 optimizer选取 2.4 train目的的选取 2.5 正式初始化tensorflow
3 tf使用session 来控制整个流程
'''



# generate some data
x_data =np.random.rand(100).astype(np.float32)
y_data = x_data*0.1+0.3

# create tensorlow structure
# initial weight and biases
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases
loss= tf.reduce_mean(tf.square(y-y_data))
## 0.5 means learning rate,choosing a optimizer. then train it
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
## initial tf
init = tf.initialize_all_variables()
##tensorflow structure end

sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step %20 ==0:
        print(step,sess.run(Weights),sess.run(biases))

