import tensorflow as tf
import pylab
#取数据
# download url  https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
import cifar10
from cifar10 import cifar10_input

batch_size = 12
data_dir = 'cifar-10-batches-bin'
images_test, labels_test = cifar10_input.inputs(eval_data = True, data_dir = data_dir, batch_size = batch_size)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()
tf.train.start_queue_runners()
image_batch, label_batch = sess.run([images_test, labels_test])
print("__\n",image_batch[0])
print("__\n",label_batch[0])
pylab.imshow(image_batch[0])
pylab.show()

