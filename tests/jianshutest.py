import tensorflow as tf
import numpy as np

from tensorflow.python.framework import  ops
import matplotlib.pyplot as plt

def test1():
    sess = tf.Session();

    a = tf.Variable([1, 2, 3], name='a')
    b = tf.constant([1, 2, 3], name='b')
    c = tf.placeholder(tf.int32, shape=[1, 2], name='myPlaceholder')

    sess.run(tf.global_variables_initializer())
    print(a);

    res = sess.run(c, feed_dict={c: [[5, 6]]})

    print(res)

    sess.close();


def test2():
    sess = tf.Session();
    a = tf.Variable(3);
    b = tf.Variable(4);

    c =tf.multiply( a, b)
    print(a)
    print(b)
    print(c)

    sess.run(tf.global_variables_initializer())
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))

    sess.close();

def test3():
    sess = tf.Session();
    a = tf.Variable(3);
    b = tf.Variable(4);

    c = tf.multiply(a, b)
    d = tf.add(a, c )
    sess.run(tf.global_variables_initializer())
    c_value = sess.run(c)
    d_value = sess.run(d);

    print( c_value, d_value)

    sess.close();

def save_model():
    sess = tf.Session();
    a = tf.Variable(5);
    b = tf.Variable(4, name="my_variable")
    op = tf.assign( a, 3)

    saver = tf.train.Saver();

    sess.run(tf.global_variables_initializer())
    sess.run(op)
    print("a", sess.run(a))
    print("my_variable", sess.run(b))
    saver.save(sess, "./my_model.ckpt")

    sess.close();


print("test1")
test1();
print("test2")
test2();
print("test3")
test3();
print("save1")
save_model();