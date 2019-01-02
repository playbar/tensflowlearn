import tensorflow as tf

input = tf.constant([[[[1], [2], [3]],
                      [[4], [5], [6]],
                      [[7], [8], [9]]]],
                    dtype=tf.float32)

# input = tf.constant([[[[-0.8113182 ], [ 1.4845988 ], [ 0.06532937]],
#                       [[-2.4427042 ], [ 0.0992484 ], [ 0.5912243 ]],
#                       [[ 0.59282297], [-2.1229296 ], [-0.72289723]]]],
#                     dtype=tf.float32)

print("input.shape")
with tf.Session() as se:
    se.run(tf.global_variables_initializer())
    print(se.run(input))

print( "filter.shape")
filter = tf.Variable(tf.random_normal([3,3,1, 1], stddev = 1, seed = 1))
# filter = tf.Variable(tf.ones([1, 1, 1, 2]));
with tf.Session() as se:
    se.run(tf.global_variables_initializer());
    print(se.run(filter));

print("tf.conv2d")
op = tf.nn.conv2d(input, filter, strides = [1, 1, 1, 1], padding = 'VALID')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(op)
    print(result)
