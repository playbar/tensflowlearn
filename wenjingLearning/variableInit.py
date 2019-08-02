# -*- coding: utf-8 -*-
import tensorflow as tf
'''
tf的定义 如果定义了variable，则必须 init 他，然后init也必须在session之中跑。
tf之中控制会话，完成操作全部需要在session之中运行

'''

# we can give variable a name
state = tf.Variable(0,name='counter')
one = tf.constant(1)

new_value = tf.add(state,one)
update =tf.assign(state,new_value)
init = tf.initialize_all_variables() # if you have declare a varible, you must initial them

with tf.Session() as sess:
    sess.run(init) # if you have init , you must put it in session and run it.
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))



