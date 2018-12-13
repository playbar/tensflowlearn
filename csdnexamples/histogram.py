#!/usr/bin/python
# coding:utf-8


# tf.summary.histogram
# tf.summary.histogram():
# 输出一个直方图的Summary protocol buffer .
#
# name：生成的节点名称.作为TensorBoard中的一个系列名称.
# values：一个实数张量.用于构建直方图的值.
# collections：图形集合键的可选列表.添加新的summary操作到这些集合中.默认为GraphKeys.SUMMARIES.
# family： summary标签名称的前缀,用于在Tensorboard上显示的标签名称.(可选项)
# tf.summary.histogram（）将输入的一个任意大小和形状的张量压缩成一个由宽度和数量组成的直方图数据结构．假设输入 [0.5, 1.1, 1.3, 2.2, 2.9, 2.99]，则可以创建三个bin，分别包含0-1之间／1-2之间／2-3之间的所有元素，即三个bin中的元素分别为［0.5］／［1.1，1.3］／［2.2，2.9，2.99］．
# 这样，通过可视化张量在不同时间点的直方图来显示某些分布随时间变化的情况．
#
# 通过在终端输入
# tensorboard --logdir=/tmp/histogram_example,

# TensorBoard直方图仪表板
import tensorflow as tf
k = tf.placeholder(tf.float32)

# # 创建一个均值变化的正态分布（由0到5左右）
#
# mean_moving_normal = tf.random_normal(shape=[1000], mean=(5*k), stddev=1)
# # 将该分布记录到直方图汇总中
# tf.summary.histogram("normal/moving_mean", mean_moving_normal)
# sess = tf.Session()
# writer = tf.summary.FileWriter("/tmp/histogram_example")
# summaries = tf.summary.merge_all()
# # 设置一个循环并将摘要写入磁盘
# N = 400
# for step in range(N):
#     k_val = step/float(N)
#     print(k_val)
#     summ = sess.run(summaries, feed_dict={k: k_val})
#     writer.add_summary(summ, global_step=step)
#
#
# print('success - 1')


k = tf.placeholder(tf.float32)
# 创建一个均值变化的正态分布
mean_moving_normal = tf.random_normal(shape=[1000], mean=(5*k), stddev=1)
# 将该分布记录到直方图summary中
tf.summary.histogram("normal/moving_mean", mean_moving_normal)
# 创建一个方差递减的正态分布
variance_shrinking_normal = tf.random_normal(shape=[1000], mean=0, stddev=1-(k))
# 记录分配
tf.summary.histogram("normal/shrinking_variance", variance_shrinking_normal)
# 将两种分布组合成一个数据集
normal_combined = tf.concat([mean_moving_normal, variance_shrinking_normal], 0)
# 添加另一个直方图summary来记录组合分布
tf.summary.histogram("normal/bimodal", normal_combined)
summaries1 = tf.summary.merge_all()
# 设置会话和摘要作者
sess1 = tf.Session()
writer = tf.summary.FileWriter("/tmp/histogram_example")
# 设置一个循环并将摘要写入磁盘
N = 400
for step in range(N):
    k_val = step/float(N)
    summ1 = sess1.run(summaries1, feed_dict={k: k_val})
    writer.add_summary(summ1, global_step=step)


print('success - 2')