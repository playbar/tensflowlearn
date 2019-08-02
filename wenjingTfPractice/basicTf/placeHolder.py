import tensorflow as tf

inputA =tf.placeholder(tf.int32)
inputB =tf.placeholder(tf.int32)
mul = tf.multiply(inputA,inputB)

with tf.Session() as sess:
    #run 永远是一个tensor ，tensor可以是一个行为，是一个计算的结果
    result = sess.run(mul,feed_dict={inputA:[1],inputB:[2]})
    print(result)
