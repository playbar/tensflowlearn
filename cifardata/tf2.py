import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

num_point = 1000;
vector_set = [];
for i in range(num_point):
    x1 = np.random.normal(0.0, 0.55);
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.3);

    vector_set.append([x1, y1]);

x_data = [v[0] for v in vector_set ];
y_data = [v[1] for v in vector_set ];

plt.scatter(x_data, y_data, c = '#FF0098')
plt.show();


w = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name = "W");
b = tf.Variable(tf.zeros([1]), name = 'b');
y = w * x_data + b;

loss = tf.reduce_mean(tf.square(y - y_data), name = "loss");
optimizer = tf.train.GradientDescentOptimizer(0.5);
train = optimizer.minimize(loss, name="train");

sess = tf.Session();
init = tf.global_variables_initializer();
sess.run(init);

for step in range(20):
    sess.run(train);
    print("W=", sess.run(w), "b=", sess.run(b), "loss=", sess.run(loss));

plt.scatter(x_data, y_data, c='r');
plt.plot(x_data, sess.run(w) * x_data +sess.run(b));
plt.show();

