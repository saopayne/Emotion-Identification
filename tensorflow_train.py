import tensorflow as tf
import data as d
import numpy as np

# data = np.genfromtxt('kddtrain_2class_normalized.csv',delimiter=',',skip_header=True)


def train(data,labels,n,c):
    x = tf.placeholder(tf.float32, [None, n])
    W = tf.Variable(tf.zeros([n, c]))
    b = tf.Variable(tf.zeros([c]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, c])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y) + (1 - y_) * tf.log(1 - y))
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(init)
    for i in range(3000):
      batch_xs, batch_ys = d.train_batch_data(data,labels,100)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    saver.save(sess,'model.ckpt')
