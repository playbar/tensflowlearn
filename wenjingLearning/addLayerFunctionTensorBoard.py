# -*- coding= utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_function = None):
    with tf.name_scope('inputs'):
        with tf.name_scope('weight'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
        with tf.name_scope('bias'):
            Baises = tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
        with tf.name_scope('wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs,Weights),Baises)
        if activation_function is None:
            outputs =Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs


x_data= np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)-0.5+noise


# define placeholder for inputs to network
# numpy shape means  改变成什么样的数组
# 把名字放入在name_scope，如果要用tensorboard的话
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1],name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1],name='y_input')

#add hidden layer
l1 =add_layer(xs,1,10,activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1,10,1,activation_function=None)

# tf里面的求loss fucntion的 reduce_mean/ sum 表示求和 求平均
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
#表示训练的目的
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)



sess = tf.Session()
writer = tf.summary.FileWriter("logs/",sess.graph)
init = tf.initialize_all_variables()
sess.run(init)