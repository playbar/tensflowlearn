import matplotlib.pyplot as plt
import tensorflow as tf

#
# tf.train.natural_exp_decay() 　应用自然指数衰减的学习率．
# learning_rate：初始学习率．
# global_step：用于衰减计算的全局步数，非负.
# decay_steps：衰减步数．
# decay_rate：衰减率．
# staircase：若为True，则是离散的阶梯型衰减（就是在一段时间内或相同的eproch内保持相同的学习率）；若为False，则是标准型衰减．
# name: 操作的名称，默认为ExponentialTimeDecay
#
# natural_exp_decay 和 exponential_decay 形式近似，natural_exp_decay的底数是e．自然指数衰减比指数衰减要快的多，一般用于较快收敛，容易训练的网络．
# 自然指数衰减的学习率计算公式为：
# decayed_learning_rate = learning_rate * exp(-decay_rate * global_step)

global_step = tf.Variable(0, name='global_step', trainable=False)
y = []
z = []
w = []
N = 200
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for global_step in range(N):
        # 阶梯型衰减
        learing_rate1 = tf.train.natural_exp_decay(
            learning_rate=0.5, global_step=global_step, decay_steps=10, decay_rate=0.9, staircase=True)
        # 标准指数型衰减
        learing_rate2 = tf.train.natural_exp_decay(
            learning_rate=0.5, global_step=global_step, decay_steps=10, decay_rate=0.9, staircase=False)
        # 指数衰减
        learing_rate3 = tf.train.exponential_decay(
            learning_rate=0.5, global_step=global_step, decay_steps=10, decay_rate=0.9, staircase=False)
        lr1 = sess.run([learing_rate1])
        lr2 = sess.run([learing_rate2])
        lr3 = sess.run([learing_rate3])
        y.append(lr1[0])
        z.append(lr2[0])
        w.append(lr3[0])

x = range(N)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim([0, 0.55])
plt.plot(x, y, 'r-', linewidth=2)
plt.plot(x, z, 'g-', linewidth=2)
plt.plot(x, w, 'b-', linewidth=2)
plt.title('natural_exp_decay')
ax.set_xlabel('step')
ax.set_ylabel('learing rate')
plt.show()

print('\n')

# tf.train.piecewise_constant()　指定间隔的分段常数．
# 参数：
# x：0-D标量Tensor．
# boundaries：边界，tensor或list.
# values：指定定义区间的值．
# name：操作的名称，默认为PiecewiseConstant．

global_step = tf.Variable(0, name='global_step', trainable=False);
boundaries = [5, 10, 15];
learing_rates = [0.1, 0.07, 0.025, 0.0125]
y = [];
N = 40

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for global_step in range(N):
        learing_rate = tf.train.piecewise_constant(global_step, boundaries=boundaries, values=learing_rates)
        lr = sess.run([learing_rate])
        y.append(lr[0])

x =range(N)
plt.plot(x, y, 'r-', linewidth=2)
plt.title('piecewise_constant')
plt.show();


print('\n');

# tf.train.exponential_decay()　应用指数衰减的学习率．指数衰减是最常用的衰减方法．
# 参数：
#
# learning_rate：初始学习率．
# global_step：用于衰减计算的全局步数，非负．用于逐步计算衰减指数．
# decay_steps：衰减步数，必须是正值．决定衰减周期．
# decay_rate：衰减率．
# staircase：若为True，则以不连续的间隔衰减学习速率即阶梯型衰减（就是在一段时间内或相同的eproch内保持相同的学习率）；若为False，则是标准指数型衰减．
# name：操作的名称，默认为ExponentialDecay．（可选项）
#
# 指数衰减的学习速率计算公式为：
# decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)

global_step = tf.Variable(0, name='global_step', trainable=False);

y = [];
z = [];
N = 200
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer());
    for global_step in range(N):
        # print(global_step );
        # 阶梯型衰减
        learing_rate1 = tf.train.exponential_decay(
            learning_rate=1.5, global_step=global_step, decay_steps=5, decay_rate=0.95, staircase=True)
        # 标准指数型衰减
        learing_rate2 = tf.train.exponential_decay(
            learning_rate=1.5, global_step=global_step, decay_steps=5, decay_rate=0.95, staircase=False)
        lr1 = sess.run([learing_rate1])
        lr2 = sess.run([learing_rate2])
        y.append(lr1[0])
        z.append(lr2[0])


x = range(N)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim([0, 2.0])
plt.plot(x, y, 'r-', linewidth=2)
plt.plot(x, z, 'g-', linewidth=2)
plt.title('exponential_decay')
ax.set_xlabel('step')
ax.set_ylabel('learing rate')
plt.show()

