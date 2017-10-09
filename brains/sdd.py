import tensorflow as tf
import numpy as np

tf.InteractiveSession()

a = tf.constant(2)
b = tf.constant(np.random.normal(size=(3,4)))
print(b.eval())

print(tf.multinomial(b,5).eval())

print(tf.random_gamma([10],[5,15]).eval())



''' sess = tf.Session()

sess.run(x)

writer = tf.summary.FileWriter('./graphs',sess.graph) '''