#!/usr/bin/python
# coding:utf-8

#  tensorboard --logdir='/tmp/logs'

import tensorflow as tf
# 迭代计数器
global_step = tf.Variable(0, trainable=False)
# 迭代+1
increment_op = tf.assign_add(global_step, tf.constant(1))
# 创建一个根据计数器衰减的Tensor
learning_rate = tf.train.exponential_decay(0.1, global_step, decay_steps=1, decay_rate=0.9, staircase=False)
# 把learning_rate添加到观测中
tf.summary.scalar('learning_rate', learning_rate)
# 获取所有检测的操作
sum_ops = tf.summary.merge_all()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # 指定检测结果的输出目录
    summary_writer = tf.summary.FileWriter('/tmp/logs/', sess.graph)
    for step in range(0, 10):
        val = sess.run(sum_ops)
        # 写入文件
        summary_writer.add_summary(val, global_step=step)
        sess.run(increment_op)


print('seccess');
