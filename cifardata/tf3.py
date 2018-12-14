import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data;

# import tflearn.datasets.mnist as lmnist
# lmnist.load_data(one_hot=True)

mnist = input_data.read_data_sets('mnist/', one_hot= True);
print(mnist.train.num_examples);
print(mnist.test.num_examples);

trainimg = mnist.train.images;
trainlabel = mnist.train.images;
testimg = mnist.test.images;
testlabel = mnist.test.labels;

print( type(trainimg));
print(trainimg.shape,);
print(trainlabel.shape,);
print(testimg.shape,);
print(testlabel.shape,);

nsample = 5;
randidx = np.random.randint(trainimg.shape[0], size = nsample )


for i in randidx:
    print(i)
    curr_img = np.reshape(trainimg[i, :], (28, 28))
    curr_label = np.argmax(trainlabel[i, :])
    plt.matshow(curr_img, cmap=plt.get_cmap('gray_r'))
    plt.title("" + str(i) + "th Training Data" + "label is" + str(curr_label))
    print("" + str(i) + "th Training Data" + "label is" + str(curr_label))
    plt.show()



