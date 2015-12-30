import csv
import untangle as xml
import numpy as np
import lsa
import re
import svmutil


def feature(terms):
    dataMatrix = np.genfromtxt(finaltest, delimiter='|', dtype=None, skip_header=True)
    n = dataMatrix.size
    l = len(terms)
    occurence = np.zeros((l, n), dtype=np.int)
    d = 0
    for row in dataMatrix:
        temp = row[0].lower().decode('UTF-8').split(' ')
        for i in range(l):
            if terms[i] in temp:
                occurence[i][d] = 1
        d += 1

    p = [i for i, e in enumerate(occurence[0]) if e != 0]
    U_, V_ = lsa.compute(occurence, 100)
    V_ = np.transpose(V_)
    return V_, dataMatrix


def dataset():
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


def predict(V, yy):
    m = svmutil.svm_load_model('sample.model')
    x = ([list(map(lambda z: z * 10000, list(t))) for t in V])
    y = [1 if t > 0 else 0 for t in yy]

    p_label, p_acc, p_val = svmutil.svm_predict(y, x, m, '')
    print(p_label)


testdata = 'AffectiveText.Semeval.2007/AffectiveText.test/affectivetext_test.xml'
testemo = 'AffectiveText.Semeval.2007/AffectiveText.test/affectivetext_test.emotions.gold'
testvalence = 'AffectiveText.Semeval.2007/AffectiveText.test/affectivetext_test.valence.gold'
finaltest = 'AffectiveText.Semeval.2007/AffectiveText.test/finalTest.txt'
