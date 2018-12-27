import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from imagenet_classes import class_names
import cv2

# 卷积层
def convLayer(x, name, kh, kw, n_out, dh, dw, p):
    # 输入数据的通道数
    n_in = x.get_shape()[-1].value
    # 设置scope
    with tf.name_scope(name) as scope:
        # 卷积核参数
        kernel = tf.Variable(tf.truncated_normal([kh, kw, n_in, n_out], dtype=tf.float32, stddev=1e-1), name='weights')
        # 卷积处理
        conv = tf.nn.conv2d(x, kernel, (1, dh, dw, 1), padding='SAME')
        # biases：初始化
        bias_init = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init, trainable=True, name='biases')
        # conv + biases
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        # 将本层参数kernel和biases存入参数列表
        p += [kernel, biases]
        return activation, p


class VGGNet16:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.convlayers()
        self.fc_layers()
        self.probs = tf.nn.softmax(self.fc3l)
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)

    def convlayers(self):
        self.parameters = []
        # 图像预处理：将RGB图像的像素值的范围设置为0-255,然后减去平均图像值
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean
        # conv1_1
        self.conv1_1, self.parameters =convLayer(images, name='conv1_1',
                                    kh=3, kw=3, n_out=64, dh=1, dw=1, p=self.parameters)
        # conv1_2
        self.conv1_2, self.parameters =convLayer(self.conv1_1, name='conv1_2',
                                    kh=3, kw=3, n_out=64, dh=1, dw=1, p=self.parameters)
        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        # conv2_1
        self.conv2_1, self.parameters =convLayer(self.pool1, name='conv2_1',
                                    kh=3, kw=3, n_out=128, dh=1, dw=1, p=self.parameters)
        # conv2_2
        self.conv2_2, self.parameters =convLayer(self.conv2_1, name='conv2_2',
                                    kh=3, kw=3, n_out=128, dh=1, dw=1, p=self.parameters)
        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='SAME', name='pool2')
        # conv3_1
        self.conv3_1, self.parameters =convLayer(self.pool2, name='conv3_1',
                                    kh=3, kw=3, n_out=256, dh=1, dw=1, p=self.parameters)
        # conv3_2
        self.conv3_2, self.parameters =convLayer(self.conv3_1, name='conv3_2',
                                    kh=3, kw=3, n_out=256, dh=1, dw=1, p=self.parameters)
        # conv3_3
        self.conv3_3, self.parameters =convLayer(self.conv3_2, name='conv3_3',
                                    kh=3, kw=3, n_out=256, dh=1, dw=1, p=self.parameters)
        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3, ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],padding='SAME', name='pool3')
        # conv4_1
        self.conv4_1, self.parameters =convLayer(self.pool3, name='conv4_1',
                                    kh=3, kw=3, n_out=512, dh=1, dw=1, p=self.parameters)
        # conv4_2
        self.conv4_2, self.parameters =convLayer(self.conv4_1, name='conv4_2',
                                    kh=3, kw=3, n_out=512, dh=1, dw=1, p=self.parameters)
        # conv4_3
        self.conv4_3, self.parameters =convLayer(self.conv4_2, name='conv4_3',
                                    kh=3, kw=3, n_out=512, dh=1, dw=1, p=self.parameters)
        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='SAME', name='pool4')
        # conv5_1
        self.conv5_1, self.parameters =convLayer(self.pool4, name='conv5_1',
                                    kh=3, kw=3, n_out=512, dh=1, dw=1, p=self.parameters)
        # conv5_2
        self.conv5_2, self.parameters =convLayer(self.conv5_1, name='conv5_2',
                                    kh=3, kw=3, n_out=512, dh=1, dw=1, p=self.parameters)
        # conv5_3
        self.conv5_3, self.parameters =convLayer(self.conv5_2, name='conv5_3',
                                    kh=3, kw=3, n_out=512, dh=1, dw=1, p=self.parameters)
        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],padding='SAME',name='pool4')

    def fc_layers(self):
        # 全连接层fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096], dtype=tf.float32,stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # 全连接层fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096], dtype=tf.float32, stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        # 全连接层fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, 1000], dtype=tf.float32, stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32), trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]
    # 加载训练好的文件模型
    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print (i, k, np.shape(weights[k]))
            sess.run(self.parameters[i].assign(weights[k]))

if __name__ == '__main__':
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    # VGGNet16模型
    vgg = VGGNet16(imgs, 'vgg16_weights.npz', sess)
    img = cv2.imread('car.jpg')
    # img = cv2.imread('crane.png')
    img1 = imresize(img, (224, 224))
    prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
    # 输出概率最高的前5种类别,以及对应的概率大小
    preds = (np.argsort(prob)[::-1])[0:5]
    for p in preds:
        print (class_names[p], prob[p])
    # 概率最大的类
    res = class_names[preds[0]]
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 显示类的名字
    cv2.putText(img, res, (int(img.shape[0] / 10), int(img.shape[1] / 10)), font, 1, (255, 0, 0), 2)
    cv2.imshow("test", img)
    cv2.waitKey(0)
