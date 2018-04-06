import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

# Creates a graph.
start = time.time()
with tf.device('/cpu:0'):
  a = tf.random_normal([3000, 9000], mean=-1, stddev=4)
  #a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.random_normal([9000, 6000], mean=-1, stddev=4)
  #b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
print('Job finished in: ' + str(time.time() - start) + ' Seconds')
