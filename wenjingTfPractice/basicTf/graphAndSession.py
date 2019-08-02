# -*- coding: utf-8 -*-
import tensorflow as tf

# 如果调用了 graph ，必须在session中指定某个graph
# 如果没有调用graph，就表示是默认的graph，则session 也可以默认
# building the graph
graph = tf.Graph()
with graph.as_default():
#     constrcut graph
    mat1 = tf.constant([1,2],name="mat1")
    mat2 = tf.constant([2,3],name="mat2")
    s = tf.add(mat1,mat2)

with tf.Session(graph=graph) as sess:
    result = sess.run(s)
    print("result:",result)

