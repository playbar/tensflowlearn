import tensorflow as tf
import numpy as np
import random

#tensorboard --logdir=/tmp/summary

a = tf.Variable( 5, name='a')
b = tf.Variable(10, name='b')
init_value = tf.multiply( a, b, name='result')

c = tf.Variable(init_value )

number = tf.placeholder( tf.int32, shape=[], name='number')
c_updata = tf.assign(c, tf.add( c, number))

tf.summary.scalar("ChangingNumber", c )

summary_op = tf.summary.merge_all();

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer());

    fw = tf.summary.FileWriter("/tmp/summary", sess.graph);

    for step in range( 1000):
        num = int( random.random() * 100 )
        sess.run(c_updata, feed_dict={number:num})

        summary = sess.run(summary_op)
        fw.add_summary(summary, step );

print("done")