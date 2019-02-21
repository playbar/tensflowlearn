import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


tf.reset_default_graph()
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True )

with tf.name_scope( 'input') as scope:
    x = tf.placeholder(tf.float32, [None, 28 * 28 ], name='input')
    labels = tf.placeholder(tf.float32, [None, 10], name='label')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

with tf.name_scope('model') as scope:
    with tf.name_scope('fc1') as scope:
        w1 = tf.Variable(tf.truncated_normal([28*28, 500], stddev=0.1), name='weights')
        b1 = tf.Variable(tf.constant( 0.1, shape=[500]), name='biases')

        with tf.name_scope('softmax_activation') as scope:
            a1 = tf.nn.softmax(tf.matmul(x, w1) + b1 )

        with tf.name_scope('dropout') as scope:
            drop1 = tf.nn.dropout(a1, keep_prob)

    with tf.name_scope('fc1') as scope:
        w2 = tf.Variable(tf.truncated_normal([500, 100], stddev=0.1), name='weights')
        b2 = tf.Variable(tf.constant(0.1, shape=[100]), name='biases')

        with tf.name_scope('relu_activation') as scope:
            a2 = tf.nn.relu(tf.matmul(drop1, w2) + b2 )

        with tf.name_scope('dropout') as scope:
            drop2 = tf.nn.dropout( a2, keep_prob )

    with tf.name_scope('fc3') as scope:
        w3 = tf.Variable(tf.truncated_normal([100, 10], stddev=0.1), name='weights')
        b3 = tf.Variable(tf.constant(0.1, shape=[10]), name='biases')

        with tf.name_scope('logits') as scope:
            logits = tf.matmul(drop2, w3) + b3;

with tf.name_scope('train') as scope:
    with tf.name_scope('loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

with tf.name_scope('evaluation') as scope:
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuray = tf.reduce_mean( tf.cast(cross_entropy, tf.float32 ))

tf.summary.scalar('Accuracy', accuray)

tf.summary.scalar('Loss', tf.reduce_mean(cross_entropy))

summary_op = tf.summary.merge_all()

saver = tf.train.Saver();


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    fw = tf.summary.FileWriter('/tmp/nn/summary', sess.graph);

    for step in range( 20000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x:batch_xs, labels:batch_ys, keep_prob:0.2})

        if step % 1000 == 0:
            acc = sess.run(accuray, feed_dict={x:batch_xs, labels:batch_ys, keep_prob:1})
            print('mid train accuracy:', acc, 'at step:', step )

        if step % 100 == 0:
            summary = sess.run(summary_op, feed_dict={x:mnist.test.images, labels: mnist.test.labels, keep_prob:1})

            fw.add_summary(summary, step)

    print("Final Test Accuracy:", sess.run(accuray, feed_dict={x:mnist.test.images, labels: mnist.test.labels, keep_prob:1}))

    saver.save(sess, '/tmp/nn/my_nn.ckpt')


print('done')