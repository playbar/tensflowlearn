import tensorflow as tf

logits = tf.constant([[1.0, 2.0, 2.0], [3.0, 1.0, 1.0], [3.0, 2.0, 1.0]]);
y = tf.nn.softmax(logits)

labels = tf.constant([[0, 1.0, 0.0], [0, 0, 1.0], [1.0, 0.0, 0]]);
cross_entropy = -tf.reduce_sum(tf.multiply(labels, tf.log(y)))

with tf.Session() as sess:
    cross_entropy_value = sess.run(cross_entropy);
    print( "cross_entropy_value=%s\n" %(cross_entropy_value))