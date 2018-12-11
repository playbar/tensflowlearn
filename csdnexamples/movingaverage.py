import tensorflow as tf

v1 = tf.Variable([1, 2, 0], dtype=tf.float32)
step = tf.Variable(tf.constant(0));

ema = tf.train.ExponentialMovingAverage( decay= 0.99, num_updates=step);

maintrain_average = ema.apply([v1]);

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer());
    print(sess.run([v1, ema.average(v1)]));

    sess.run(tf.assign(v1, [1, 2, 5]));
    sess.run(maintrain_average);

    print(sess.run([v1, ema.average(v1)]));

    sess.run(tf.assign(step, 10000));

    sess.run(tf.assign(v1, [1, 3, 10]));
    sess.run(maintrain_average);

    print(sess.run([v1, ema.average(v1)]))
    sess.run(maintrain_average)
    print(sess.run([v1, ema.average(v1)]))

