import tensorflow as tf
t = [1,2,3,4,5,6,7,8,9]
x = tf.strided_slice(t,[0],[5], [3])
y = tf.strided_slice(t,[1],[-5])
with tf.Session() as sess:
    print(sess.run(x))
    print(sess.run(y))

print("========\n")

t = tf.constant([[[1, 1, 1], [2, 2, 2], [7, 7, 7]],
                 [[3, 3, 3], [4, 4, 4], [8, 8, 8]],
                 [[5, 5, 5], [6, 6, 6], [9, 9, 9]]])

z1 = tf.strided_slice(t, [1], [-1], [1])
z2 = tf.strided_slice(t, [1, 0], [-1, 2], [1, 1])
z3 = tf.strided_slice(t, [1, 0, 1], [-1, 2, 3], [1, 1, 1])

with tf.Session() as sess:
    print(sess.run(z1))
    print(sess.run(z2))
    print(sess.run(z3))
