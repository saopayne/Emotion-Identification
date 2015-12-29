import csv
import urllib.request as get
import numpy as np
import untangle as xml
import lsa
import tensorflow as tf

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
    occurence = np.zeros((l, n),dtype=np.int)
    d = 0
    for row in dataMatrix:
        temp = row[0].decode('UTF-8').split(' ')
        for i in range(l):
            if terms[i] in temp:
                occurence[i][d] = 1
        d += 1

    p = [i for i,e in enumerate(occurence[0]) if e!=0]
    U_,V_ = lsa.compute(occurence,50)
    V_ = np.transpose(V_)
    return V_



def train(V,c):
    n = V.shape[1]
    x = tf.placeholder(tf.float32,[None,n])
    W = tf.Variable(tf.zeros([n, c]))
    b = tf.Variable(tf.zeros([c]))
    y = tf.nn.softmax(tf.matmul(x,W)+b)





trialdata = 'AffectiveText.Semeval.2007/AffectiveText.trial/affectivetext_trial.xml'
trialvalence = 'AffectiveText.Semeval.2007/AffectiveText.trial/affectivetext_trial.valence.gold'
testdata = 'AffectiveText.Semeval.2007/AffectiveText.test/affectivetext_test.xml'
trialemo = 'AffectiveText.Semeval.2007/AffectiveText.trial/affectivetext_trial.emotions.gold'
finaltrial = 'AffectiveText.Semeval.2007/AffectiveText.trial/finalTrial.txt'

obj = xml.parse(trialdata)
target = open(finaltrial, 'r+')
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
M = feature()
train(M,6)
