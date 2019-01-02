import tensorflow as tf




# tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
# 除去name参数用以指定该操作的name，与方法有关的一共五个参数：
#
# 第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]
# 这样的shape，具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个4维的Tensor，
# 要求类型为float32和float64其中之一
#
# 第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]
# 这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，
# 第三维in_channels，就是参数input的第四维
#
# 第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
#
# 第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式（后面会介绍）
#
# 第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true
#
# 结果返回一个Tensor，这个输出，就是我们常说的feature map

################
# 1.考虑一种最简单的情况，现在有一张3×3单通道的图像（对应的shape：[1，3，3，1]），
# 用一个1×1的卷积核（对应的shape：[1，1，1，1]）去做卷积，最后会得到一张3×3的feature map

input = tf.Variable(tf.random_normal([1,3,3,1], stddev = 1, seed = 1))
filter = tf.Variable(tf.random_normal([1,1,1,1]))
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
with tf.Session() as a_sess:
    a_sess.run(tf.global_variables_initializer());
    res = (a_sess.run(op));
    print("res.shape");
    print(res.shape);
    print(res)


# 2.增加图片的通道数，使用一张3×3五通道的图像（对应的shape：[1，3，3，5]），
# 用一个1×1的卷积核（对应的shape：[1，1，1，1]）去做卷积，仍然是一张3×3的feature map，
# 这就相当于每一个像素点，卷积核都与该像素点的每一个通道做卷积。

input = tf.Variable(tf.random_normal([1,3,3,1], stddev = 1, seed = 1))
filter = tf.Variable(tf.random_normal([1,1,1,2]))
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
with tf.Session() as a_sess:
    a_sess.run(tf.global_variables_initializer());
    print(a_sess.run(op));

oplist=[]

input_arg = tf.Variable(tf.random_normal([1, 3, 3, 1]));
filter_arg = tf.Variable(tf.random_normal([1, 1, 1, 2]));
op2 = tf.nn.conv2d(input_arg, filter_arg, strides=[1,1,1,1], use_cudnn_on_gpu=False, padding='VALID')
oplist.append([op2, "case 1"])

with tf.Session() as a_sess:
    a_sess.run(tf.global_variables_initializer());
    print(a_sess.run(input_arg.initializer));
    print(a_sess.run(op2));

# [batch, in_height, in_width, in_channels]
input_arg  = tf.Variable(tf.ones([1, 3, 3, 2]))
# [filter_height, filter_width, in_channels, out_channels]
filter_arg = tf.Variable(tf.ones([1 ,1 , 2, 6 ]))

op2 = tf.nn.conv2d(input_arg, filter_arg, strides=[1,1,1,1], use_cudnn_on_gpu=False, padding='VALID')
oplist.append([op2, "case 2"])

with tf.Session() as a_sess:
    a_sess.run(tf.global_variables_initializer());
    print(input_arg);
    print(a_sess.run(op2));

# [batch, in_height, in_width, in_channels]
input_arg  = tf.Variable(tf.ones([1, 3, 3, 5]))
# [filter_height, filter_width, in_channels, out_channels]
filter_arg = tf.Variable(tf.ones([3 ,3 , 5 ,1]))

op2 = tf.nn.conv2d(input_arg, filter_arg, strides=[1,1,1,1], use_cudnn_on_gpu=False, padding='VALID')
oplist.append([op2, "case 3"])

# [batch, in_height, in_width, in_channels]
input_arg  = tf.Variable(tf.ones([1, 5, 5, 5]))
# [filter_height, filter_width, in_channels, out_channels]
filter_arg = tf.Variable(tf.ones([3 ,3 , 5 ,1]))

op2 = tf.nn.conv2d(input_arg, filter_arg, strides=[1,1,1,1], use_cudnn_on_gpu=False, padding='VALID')
oplist.append([op2, "case 4"])

# [batch, in_height, in_width, in_channels]
input_arg  = tf.Variable(tf.ones([1, 5, 5, 5]))
# [filter_height, filter_width, in_channels, out_channels]
filter_arg = tf.Variable(tf.ones([3 ,3 , 5 ,1]))
op2 = tf.nn.conv2d(input_arg, filter_arg, strides=[1,1,1,1], use_cudnn_on_gpu=False, padding='SAME')
oplist.append([op2, "case 5"])

# [batch, in_height, in_width, in_channels]
input_arg  = tf.Variable(tf.ones([1, 5, 5, 5]))
# [filter_height, filter_width, in_channels, out_channels]
filter_arg = tf.Variable(tf.ones([3 ,3 , 5 ,7]))
op2 = tf.nn.conv2d(input_arg, filter_arg, strides=[1,1,1,1], use_cudnn_on_gpu=False, padding='SAME')
oplist.append([op2, "case 6"])


# [batch, in_height, in_width, in_channels]
input_arg  = tf.Variable(tf.ones([1, 5, 5, 5]))
# [filter_height, filter_width, in_channels, out_channels]
filter_arg = tf.Variable(tf.ones([3 ,3 , 5 ,7]))
op2 = tf.nn.conv2d(input_arg, filter_arg, strides=[1,2,2,1], use_cudnn_on_gpu=False, padding='SAME')
oplist.append([op2, "case 7"])


# [batch, in_height, in_width, in_channels]
input_arg  = tf.Variable(tf.ones([4, 5, 5, 5]))
# [filter_height, filter_width, in_channels, out_channels]
filter_arg = tf.Variable(tf.ones([3 ,3 , 5 ,7]))
op2 = tf.nn.conv2d(input_arg, filter_arg, strides=[1,2,2,1], use_cudnn_on_gpu=False, padding='SAME')
oplist.append([op2, "case 8"])

with tf.Session() as a_sess:
    a_sess.run(tf.global_variables_initializer())
    for aop in oplist:
        print("----------{}---------".format(aop[1]))
        print(a_sess.run(aop[0]))
        print('---------------------\n\n')

print("success");