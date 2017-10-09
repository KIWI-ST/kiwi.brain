import tensorflow as tf

#modle
W = tf.Variable([.3],dtype=tf.float32)
b = tf.Variable([-.3],dtype=tf.float32)

#input
x = tf.placeholder(tf.float32)
#formula y = ax+b
linear_model = W*x+b
#output
y = tf.placeholder(tf.float32)
#loss
loss = tf.reduce_sum(tf.square(linear_model-y))
#optimizer 
optimizer = tf.train.GradientDescentOptimizer(0.001)


train = optimizer.minimize(loss)

#training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
#training loop
init = tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
for i in range(10000):
    sess.run(train,{x:x_train,y:y_train})
    curr_w,curr_b,curr_loss = sess.run([W,b,loss],{x:x_train,y:y_train})
    print("I:%s W: %s b: %s loss: %s"%(i,curr_w, curr_b, curr_loss))