import tensorflow as tf
import numpy as np
import data as d
from collections import Counter
import copy

n = 40
c = 5
def display(actual, predicted):
    r = dict.fromkeys(set(actual), 0)
    n = copy.deepcopy(r)
    for x, y in zip(actual, predicted):
        if x == y:
            r[x] += 1
        n[x] += 1
    print(r,n)


def predict(data,labels,n,c):
    x = tf.placeholder(tf.float32, [None, n])
    W = tf.Variable(tf.zeros([n, c]))
    b = tf.Variable(tf.zeros([c]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, c])


    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess,'model.ckpt')

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    a, b = data, d.one_hot(labels)
    actual = (sess.run(tf.argmax(y_,1),feed_dict = {y_ : b}))
    predicted = (sess.run(tf.argmax(y,1),feed_dict = {x : a, y_ : b}))
    # print(predicted)
    # print(actual)
    display(actual,predicted)
    print('Accuracy = ', sess.run(accuracy, feed_dict={x: a, y_: b}))
