import csv
import urllib.request as get
import numpy as np
import untangle as xml
import lsa
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import svm
import svmutil


def feature():
    dataMatrix = np.genfromtxt(finaltrial, delimiter='\t', dtype=None, skip_header=True)
    terms = []
    n = dataMatrix.size
    for row in dataMatrix:
        row[0] = row[0].lower().decode('UTF-8')
        temp = row[0].decode('UTF-8').replace(' ', '+')
        temp = (get.urlopen("http://localhost:5095/parser?sentence=" + temp).read()).decode('UTF-8')
        terms.extend([x.split('/')[0] for x in temp.split(' ') if
                      x.split('/')[1] == 'JJ' or x.split('/')[1].startswith('VB')])

    terms = list(set(terms))
    stop = open('stop.csv', 'r').read().splitlines()
    terms = [x for x in terms if x not in stop]
    l = len(terms)
    occurence = np.zeros((l, n), dtype=np.int)
    d = 0
    for row in dataMatrix:
        temp = row[0].decode('UTF-8').split(' ')
        for i in range(l):
            if terms[i] in temp:
                occurence[i][d] = 1
        d += 1

    p = [i for i, e in enumerate(occurence[0]) if e != 0]
    U_, V_ = lsa.compute(occurence, 100)
    V_ = np.transpose(V_)
    return V_, dataMatrix

def test(terms):
    dataMatrix = np.genfromtxt(finalTest,delimiter='\t',dtype=None,skip_header=True)
    l = len(terms)
    occurence = np.zeros((l, n), dtype=np.int)
    d = 0
    for row in dataMatrix:
        temp = row[0].decode('UTF-8').split(' ')
        for i in range(l):
            if terms[i] in temp:
                occurence[i][d] = 1
        d += 1

    p = [i for i, e in enumerate(occurence[0]) if e != 0]
    U_, V_ = lsa.compute(occurence, 100)
    V_ = np.transpose(V_)
    return V_


def train(V, c, yy):
    # n = V.shape[1]
    # x = tf.placeholder(tf.float32,[None,n])
    # W = tf.Variable(tf.zeros([n, c]))
    # b = tf.Variable(tf.zeros([c]))
    # y = tf.nn.softmax(tf.matmul(x,W)+b)
    # y_ = tf.placeholder(tf.float32, [None, c])
    # cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    # train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    # init = tf.initialize_all_variables()
    # sess = tf.Session()
    # sess.run(init)
    # sess.run
    # for i in range(1000):
    #     sess.run(train_step, feed_dict={x: V, y_: yy})

    x = ([list(map(lambda z: z * 10000, list(t))) for t in V])
    y = [1 if t > 0 else 0 for t in yy]

    print(x[0])
    prob = svmutil.svm_problem(y,x)
    param = svmutil.svm_parameter('-e 0.01')
    m = svmutil.svm_train(prob,param)
    svmutil.svm_save_model('sample.model',m)
    p_label, p_acc, p_val = svmutil.svm_predict(y, x, m, '')
    print(len(y))
    return m




trialdata = 'AffectiveText.Semeval.2007/AffectiveText.trial/affectivetext_trial.xml'
trialvalence = 'AffectiveText.Semeval.2007/AffectiveText.trial/affectivetext_trial.valence.gold'
testdata = 'AffectiveText.Semeval.2007/AffectiveText.test/affectivetext_test.xml'
trialemo = 'AffectiveText.Semeval.2007/AffectiveText.trial/affectivetext_trial.emotions.gold'
testemo = 'AffectiveText.Semeval.2007/AffectiveText.test/affectivetext_test.emotions.gold'
testvalence = 'AffectiveText.Semeval.2007/AffectiveText.test/affectivetext_test.valence.gold'
finaltrial = 'AffectiveText.Semeval.2007/AffectiveText.trial/finalTrial.txt'
finaltest = 'AffectiveText.Semeval.2007/AffectiveText.test/finalTest.txt'
obj = xml.parse(trialdata)
target = open(finaltrial, 'w')
target.write('A\tanger\tdisgust\tfear\tjoy\tsadness\tsurprise\tval')
target.write('\n')
with open(trialemo, 'r') as f, open(trialvalence, 'r') as g:
    r = csv.reader(f, delimiter=' ')
    s = csv.reader(g, delimiter=' ')
    i = 0
    for row, row1 in zip(r, s):
        str = [obj.corpus.instance[i].cdata, row[1], row[2], row[3], row[4], row[5], row[6], row1[1]]
        target.write('\t'.join(str))
        target.write('\n')
        i += 1
target.close()
M, D = feature()
y = [t[6] for t in D]
train(M, 6, y)
