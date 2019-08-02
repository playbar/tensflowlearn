from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

## read 0-9 data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

## the function to compare accuracy between prediction and true value
def compute_accuracy(v_xs,v_ys):
    global  prediction
    # every time session run, keep_pro is the dropout parameter
    y_pre = sess.run(prediction ,feed_dict={xs:v_xs,keep_prob:1})
    #caculate  argmax Returns the index with the largest value across axes of a tensor.
    #predict the biggest index ... to judge if is a  correct prediction
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    #caclulate all the precdction's accurarcy, then have the result
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict ={xs:v_xs,ys:v_ys,keep_prob:1})

## generating initial weight value
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

# define the whole NN at first

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32,[None,784]) #28*28 明确了有784列数据，但是不知道有多少行
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placehoder(tf.float32)
# make the shape fit to cnn [n_samples,28,28,1]
x_image = tf.reshape(xs,[-1,28,28,1])

##conv1 layer
W_conv1 = weight_variable([5,5,1,32]) #patch size 5*5 ,in_size 1(grey picture ,so input channl is 1,we define output size 32, in this case, we have 32 patches)
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1) #using relu to filter the negitive value # output size 28*28*32
h_pool1 = max_pool_2x2(h_conv1)# using maxpooling made output --- 14*14*32

##conv2 layer
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2) # output size 14*14*64
h_pool2 = max_pool_2x2(h_conv2) #using maxingpolling made output 7*7 *64

# fc1 layer
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
#[nsample,7,7,64]--->[nsample,7*7*64]
h_poo2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_poo2_flat,W_fc1)+b_fc1)
#dropout was added on fc layer
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

# fc2 layer
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

#the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1])) # loss
# how to train? using what method to boost training
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# gpu setting
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'

# initial tf parameter
sess =tf.Session()
# initial all variable
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_x,batch_y = mnist.train.next_batch(50)
    sess.run(train_step,feed_dict={xs:batch_x,ys:batch_y,keep_prob:0.5})
    if i %25 == 0:
        print(compute_accuracy(mnist.test.images[0:4000],mnist.test.labels[0:4000]))
