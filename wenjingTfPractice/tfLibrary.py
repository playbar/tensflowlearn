import tensorflow as tf


def add_layer(inputs, in_size, out_size,activation_function = None):
    '''general function for tf to add a hidden layer '''
    # generating a fully connection weights matrix
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    # generating bias  , we need out_size bias
    Bias = tf.Variable(tf.zeros([1,out_size])+0.1)
    # wx+b
    Wx_plus_b = tf.matmul(inputs,Weights)+Bias
    if activation_function is None:
        output = Wx_plus_b
    else:
        output =activation_function(Wx_plus_b)
    return  output
