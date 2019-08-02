# -*- coding=utf-8 -*-
import tensorflow as tf
'''
placeholder 相当于 预先定义好的某一个变量类型的变量，在session run的时候传值进去，这样可以不用初始化

'''
input1 =tf.placeholder(tf.float32)
input2 =tf.placeholder(tf.float32)

output = tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))
