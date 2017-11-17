import tensorflow as tf
import numpy as np


def layer(inputs, in_size, out_size, act_func=None):
    #W matrix
    W = tf.Variable(tf.random_normal([in_size, out_size]))
    #B matrix
    B = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    #linear function
    Wx_plus_b = tf.matmul(inputs, W) + B
    if act_func is None:
        y = Wx_plus_b
    else:
        y = act_func(Wx_plus_b)
    return y

#开始构建神经网络输入层

xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

#add hidden layer
l1 = layer(xs,1,10,act_func=tf.nn.relu)
l2= layer(l1,10,1,act_func=tf.nn.sigmoid)
loss =tf.reduce_mean(tf.reduce_sum(ys * tf.log(l2),reduction_indices=[1]))

#train
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

x_data = np.linspace(-1,1,300)[:,np.newaxis] # 转为列向量 
y_data = np.square(x_data) 

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(10000):
    w,x,b = sess.run(train,feed_dict={xs:x_data,ys:y_data})
    print(w+x+b)
