import csv
import untangle as xml
import numpy as np
import re
import svmutil


def feature(terms, U):
    dataMatrix = np.genfromtxt(finaltrial, delimiter='|', dtype=None, skip_header=True)
    n = dataMatrix.size
    l = len(terms)
    occurence = np.zeros((n, l), dtype=np.int)
    d = 0
    for row in dataMatrix:
        temp = row[0].lower().decode('UTF-8').split(' ')
        for i in range(l):
            if terms[i] in temp:
                occurence[d][i] = 1
        d += 1

    occurence = np.array([np.dot(o, U) for o in occurence])
    print(occurence.shape)

    return occurence, dataMatrix


def dataset():
    obj = xml.parse(trialdata)
    target = open(finaltrial, 'w')
    target.write('A\tanger\tdisgust\tfear\tjoy\tsadness\tsurprise\tval')
    target.write('\n')
    with open(trialemo, 'r') as f, open(trialvalence, 'r') as g:
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


def predict(V, yy):
    m = svmutil.svm_load_model('sample.model')
    x = ([list(map(lambda z: z * 100, list(t))) for t in V])
    #y = [1 if t > 0 else 0 for t in yy]
    p_label, p_acc, p_val = svmutil.svm_predict(yy, x, m)
    print(yy)
    print(p_val)



trialdata = 'AffectiveText.Semeval.2007/AffectiveText.trial/affectivetext_trial.xml'
trialvalence = 'AffectiveText.Semeval.2007/AffectiveText.trial/affectivetext_trial.valence.gold'
testdata = 'AffectiveText.Semeval.2007/AffectiveText.test/affectivetext_test.xml'
trialemo = 'AffectiveText.Semeval.2007/AffectiveText.trial/affectivetext_trial.emotions.gold'
testemo = 'AffectiveText.Semeval.2007/AffectiveText.test/affectivetext_test.emotions.gold'
testvalence = 'AffectiveText.Semeval.2007/AffectiveText.test/affectivetext_test.valence.gold'
finaltrial = 'AffectiveText.Semeval.2007/AffectiveText.trial/finalTrial.txt'
finaltest = 'AffectiveText.Semeval.2007/AffectiveText.test/finalTest.txt'
