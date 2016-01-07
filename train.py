import csv
import re
import urllib.request as get
import tensorflow as tf
import numpy as np
import untangle as xml
from sklearn.feature_extraction.text import TfidfTransformer
import lsa
import svmutil, svm
import test
import data as d
import tensorflow_train as tftrain
import tensorflow_predict as tfpredict
from collections import Counter
termcount = {}

def tfidf(temp):
    global termcount
    for term in temp.split(' '):
        t = term.split('/')[0]
        if t in list(termcount.keys()):
            termcount[t] += 1
        else:
            termcount[t] = 1
    return termcount

def feature():
    global termcount
    dataMatrix = np.genfromtxt(finaltrial, delimiter='|', dtype=None, skip_header=True)
    terms = []
    n = dataMatrix.size
    for row in dataMatrix:
        row[0] = row[0].lower().decode('UTF-8')
        temp = row[0].decode('UTF-8').replace(' ', '+')
        temp = (get.urlopen("http://localhost:5095/parser?sentence=" + temp).read()).decode('UTF-8')
        terms.extend([x.split('/')[0] for x in temp.split(' ') if
                      x.split('/')[1] == 'JJ' or x.split('/')[1].startswith('VB')])
        tfidf(temp)
    s = sum(list(termcount.values()))
    termcount = {x: (y * 100 / s) for x, y in zip(termcount.keys(), termcount.values())}
    # terms.extend([x for x in termcount.keys()])
    terms = list(set(terms))
    stop = open('stop.csv', 'r').read().splitlines()
    terms = [x for x in terms if x not in stop]
    l = len(terms)
    occurence = np.zeros((n, l), dtype=np.int)
    d = 0
    for row in dataMatrix:
        temp = row[0].decode('UTF-8').split(' ')
        for i in range(l):
            if terms[i] in temp:
                occurence[d][i] += 1
        d += 1
    transformer = TfidfTransformer()
    tfdif = transformer.fit_transform(occurence)
    occurence = tfdif.toarray()


    np.savetxt('occurence.csv',occurence,delimiter=',')
    return occurence, dataMatrix, terms




trialdata = 'AffectiveText.Semeval.2007/AffectiveText.trial/affectivetext_trial.xml'
trialvalence = 'AffectiveText.Semeval.2007/AffectiveText.trial/affectivetext_trial.valence.gold'
testdata = 'AffectiveText.Semeval.2007/AffectiveText.test/affectivetext_test.xml'
trialemo = 'AffectiveText.Semeval.2007/AffectiveText.trial/affectivetext_trial.emotions.gold'
testemo = 'AffectiveText.Semeval.2007/AffectiveText.test/affectivetext_test.emotions.gold'
testvalence = 'AffectiveText.Semeval.2007/AffectiveText.test/affectivetext_test.valence.gold'
finaltrial = 'AffectiveText.Semeval.2007/AffectiveText.trial/finalTrial.txt'
finaltest = 'AffectiveText.Semeval.2007/AffectiveText.test/finalTest.txt'
obj = xml.parse(testdata)
target = open(finaltest, 'w')
target.write('A\tanger\tdisgust\tfear\tjoy\tsadness\tsurprise\tval')
target.write('\n')
with open(testemo, 'r') as f, open(testvalence, 'r') as g:
    r = csv.reader(f, delimiter=' ')
    s = csv.reader(g, delimiter=' ')
    i = 0
    for row, row1 in zip(r, s):
        temp = obj.corpus.instance[i].cdata
        temp = re.sub(r'[^\w]', ' ', temp)
        str = [temp, row[1], row[2], row[3], row[4], row[5], row[6], row1[1]]
        target.write('|'.join(str))
        target.write('\n')
        i += 1
target.close()

# driver code
V, D , Terms = feature()
y = [0 if -100 <= t[7] <= -50 else 1 if -50 < t[7] < 50 else 2 for t in D]
# tftrain.train(V,y,306,3)
# test.dataset()
# V, D = test.feature(Terms)
y = [0 if -100 <= t[7] <= -50 else 1 if -50 < t[7] < 50 else 2 for t in D]
tfpredict.predict(V, y, 306, 3)
