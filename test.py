import csv
import untangle as xml
trialdata = 'AffectiveText.Semeval.2007/AffectiveText.trial/affectivetext_trial.xml'
trialvalence = 'AffectiveText.Semeval.2007/AffectiveText.trial/affectivetext_trial.valence.gold'
trialemo = 'AffectiveText.Semeval.2007/AffectiveText.trial/affectivetext_trial.emotions.gold'
finaltrial = 'AffectiveText.Semeval.2007/AffectiveText.trial/finalTrial.txt'


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