import tensorflow as tf
'''使用TensorFlow的时候，你需要理解的一些tensorflow问题： 
怎么用图表示计算； 
在Session里面计算图； 
用tensor表示数据； 
用变量保持状态； 
用feeds（联系placeholder）和fetches来从任意的操作（Operation）中“放入”或者“拿出”数据。 
再回忆一下tensorflow的思想：首先是构造过程来“组装”一个图，然后是执行过程用session来执行图中的操作'''
v1 = tf.Variable(1,name='v1')
v2 = tf.Variable(2,name='v2')

one = tf.constant(1)
new_value =tf.add(v1,one)
update = tf.assign(v1,new_value)
#tf 需要一个 init all variable 的动作来初始化所有的varaibale
init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    # tf 要在session之中跑一次tensor，需要用run，而不是直接打印出来
    print(sess.run(v1))
    for _ in range(3):
        sess.run(update)
        print(sess.run(v1))