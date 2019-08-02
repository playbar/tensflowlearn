# -*- coding: utf-8 -*-
import tensorflow as tf

'''
这个例子是用来告诉大家如何初始化session，以及使用的，相当于python的file handle方式
同时tf之中的每一个操作都必须通过session 来 run

'''

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])

product = tf.matmul(matrix1,matrix2) #matrix multiply np.dot(m1,m2)

#method 1
# sess = tf.Session()
# result =sess.run(product)
# print(result)
# sess.close()

#method 2,with synax can help you close session convient
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)

