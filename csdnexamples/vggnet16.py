#!/usr/bin/python
# -*- coding: UTF-8 -*-

# TensorFlow实现VGGNet-16

from datetime import datetime
import math
import time
import tensorflow as tf

# 卷积层
# kh, kw：卷积核尺寸
# n_out：卷积核输出通道数
# dh, dw：步长
# p：参数列表
def convLayer(x, name, kh, kw, n_out, dh, dw, p):
    # 输入数据的通道数
    n_in = x.get_shape()[-1].value
    # 设置scope
    with tf.name_scope(name) as scope:
        # 卷积核参数
        kernel = tf.get_variable(scope+'w',shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        # 卷积处理
        conv = tf.nn.conv2d(x, kernel, (1, dh, dw, 1), padding='SAME')
        # biases：初始化
        bias_init = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init, trainable=True, name='b')
        # conv + biases
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        # 将本层参数kernel和biases存入参数列表
        p += [kernel, biases]
        return activation

# 全连接层
def fcLayer(x, name, n_out, p):
    # 输入x的通道数
    n_in = x.get_shape()[-1].value
    # 设置scope
    with tf.name_scope(name) as scope:
        #
        kernel = tf.get_variable(scope+'w',
        shape=[n_in, n_out], dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
        # 将biases初始化一个较小的值以避免dead neuron
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
        # Relu(x * kernel + biases).
        activation = tf.nn.relu_layer(x, kernel, biases, name='b')
        # 将本层参数存入参数列表
        p += [kernel, biases]
        return activation


#
def VGGNet(x, keep_prob):
    p = []
    # 卷积1　卷积核3*3,数量64,步长1*1
    # 输入224*224*3,输出224*224*64
    conv1_1 = convLayer(x, name='conv1_1', kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    # 输入224*224*64,输出224*224*64
    conv1_2 = convLayer(conv1_1, name='conv1_2', kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    # 池化尺寸2*2,步长2*2
    # 输入224*224*64,输出112*112*64
    pool1 = tf.nn.max_pool(conv1_2,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool1')

    # 卷积2　卷积核3*3,数量128,步长1*1
    # 输入112*112*64,输出112*112*128
    conv2_1 = convLayer(pool1, name='conv2_1', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    # 输入112*112*128,输出112*112*128
    conv2_2 = convLayer(conv2_1, name='conv2_2', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    # 池化尺寸2*2,步长2*2
    # 输入112*112*128,输出56*56*128
    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # 卷积3　卷积核3*3,数量256,步长1*1
    # 输入56*56*128,输出56*56*256
    conv3_1 = convLayer(pool2, name='conv3_1', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    # 输入56*56*256,输出56*56*256
    conv3_2 = convLayer(conv3_1, name='conv3_2', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    # 输入56*56*256,输出56*56*256
    conv3_3 = convLayer(conv3_2, name='conv3_3', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    # 池化尺寸2*2,步长2*2
    # 输入56*56*256,输出28*28*256
    pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    # 卷积4　卷积核3*3,数量512,步长1*1
    # 输入28*28*256,输出28*28*512
    conv4_1 = convLayer(pool3, name='conv4_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    # 输入28*28*512,输出28*28*512
    conv4_2 = convLayer(conv4_1, name='conv4_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    # 输入28*28*512,输出28*28*512
    conv4_3 = convLayer(conv4_2, name='conv4_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    # 池化尺寸2*2,步长2*2
    # 输入28*28*512,输出14*14*512
    pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

    # 卷积5　卷积核3*3,数量512,步长1*1
    # 输入14*14*512,输出14*14*512
    conv5_1 = convLayer(pool4, name='conv5_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    # 输入14*14*512,输出14*14*512
    conv5_2 = convLayer(conv5_1, name='conv5_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    # 输入14*14*512,输出14*14*512
    conv5_3 = convLayer(conv5_2, name='conv5_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    # 池化尺寸2*2,步长2*2
    # 输入14*14*512,输出7*7*512
    pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')

    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    # 将每个样本化为长度为7*7*512的一维向量
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name='resh1')

    # 全连接层　FC-4096
    fc6 = fcLayer(resh1, name='fc6', n_out=4096, p=p)
    # Dropout层训练时节点保留率为0.5
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name='fc6_drop')
    # 全连接层　FC-4096
    fc7_drop = fcLayer(fc6_drop, name='fc7', n_out=4096, p=p)
    # 全连接层　FC-1000
    fc8 = fcLayer(fc7_drop, name='fc8', n_out=1000, p=p)

    softmax = tf.nn.softmax(fc8)
    # 求出概率最大的类别
    predictions = tf.argmax(softmax, 1)
    return predictions, softmax, fc8, p


#
def time_tensorflow_run(session, target, feed, info_string):
    num_batches = 100
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target, feed_dict=feed)
        # 持续时间
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print ('%s: step %d, duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))
                # 总持续时间
                total_duration += duration
                # 总持续时间平方和
                total_duration_squared += duration * duration
    # 计算每轮迭代的平均耗时mn,和标准差sd
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    # 打印出每轮迭代耗时
    print ('%s: %s across %d steps, %.3f +/- %.3f sec /batch' % (datetime.now(), info_string, num_batches, mn, sd))



#
def run_benchmark():
    with tf.Graph().as_default():
        batch_size = 32
        image_size = 224
        # 生成尺寸为224*224的随机图片
        images = tf.Variable(tf.random_normal([batch_size,
                                               image_size,
                                               image_size,3],
                                              dtype=tf.float32,
                                              stddev=1e-1))

        keep_prob = tf.placeholder(tf.float32)
        # 构建VGGNet-16网络
        predictions, softmax, fc8, p =VGGNet(images, keep_prob)
        # 创建Session并初始化全局参数
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        # 将keep_prob置为1,进行预测predictions,测评forward运算时间
        time_tensorflow_run(sess, predictions, {keep_prob:1.0}, 'Froward')
        objective = tf.nn.l2_loss(fc8)
        grad = tf.gradients(objective, p)
        # 将keep_prob置为0.5,求解梯度操作grad,测评backward运算时间
        time_tensorflow_run(sess, grad, {keep_prob:0.5}, 'Forward-backward')


if __name__ == '__main__':
    run_benchmark()

