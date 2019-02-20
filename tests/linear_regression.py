import tensorflow as tf
import numpy as np

from tensorflow.python.framework import  ops
import matplotlib.pyplot as plt

ops.reset_default_graph();

sess = tf.Session();

data_amount = 101 # 数据数量
batch_size = 25 # 批量大小

# 造数据 y=Kx+3 (K=5)
x_vals = np.linspace(0, 200, data_amount)

# print(x_vals)

y_vals = np.multiply(x_vals, 5)
# print( y_vals )
y_vals = np.add(y_vals, 3)

# print( y_vals)

# 生成一个N(0,15)的正态分布一维数组
y_offset_vals = np.random.normal(0, 15, data_amount)
print(y_offset_vals)
y_vals = np.add(y_vals, y_offset_vals) # 为了有意使的y值有所偏差

# 创建占位符
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 构造K 就是要训练得到的值
K = tf.Variable(tf.random_normal(mean=0, shape=[1, 1]))

calcY = tf.add(tf.matmul(x_data, K), 3)

# 真实值与模型估算的差值
loss = tf.reduce_mean(tf.square(y_target - calcY))

init = tf.global_variables_initializer()
sess.run(init)

my_opt = tf.train.GradientDescentOptimizer(0.0000005)
train_step = my_opt.minimize(loss) # 目的就是使损失值最小

loss_vec = [] #保存每次迭代的损失值，为了图形化

for i in range(1000):
    rand_index = np.random.choice(data_amount, size=batch_size)
    x = np.transpose([x_vals[rand_index]])
    y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: x, y_target: y})

    tmp_loss = sess.run(loss, feed_dict={x_data: x, y_target: y})
    loss_vec.append(tmp_loss)
# 每25的倍数输出往控制台输出当前训练数据供查看进度
    if (i + 1) % 25 == 0:
        print('Step #' + str(i + 1) + ' K = ' + str(sess.run(K)))
        print('Loss = ' + str(sess.run(loss, feed_dict={x_data: x, y_target: y})))

KValue  = sess.run(K);
print( "K %d" %KValue)

best_fit = np.linspace(0, 0, data_amount)

for i in range(data_amount):
    val = KValue * x_vals[i] + 3;
    # print(val)
    best_fit[i] = val


# print(x_vals)
# print(y_vals)
# print(best_fit)

plt.plot(x_vals, y_vals, 'o', label='Data')
plt.plot(x_vals, best_fit, 'r-', label='Base fit line')
# plt.plot(loss_vec, 'k-')
plt.title('Batch Look Loss')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
# 当训练完成后k的值就是当前的得到的结果，可以通过sess.run(K)取得
sess.close()

