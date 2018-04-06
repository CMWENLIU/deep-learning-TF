import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

x = 2
y = 3
add_op = tf.add(x, y)
mul_op = tf.multiply(x, y)
output_1 = tf.multiply(x, add_op)
output_2 = tf.pow(add_op, mul_op)
with tf.Session() as sess:
  output_1, output_2 = sess.run([output_1, output_2])
print(output_1, output_2)

