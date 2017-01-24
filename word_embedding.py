# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
# numpy
import numpy
# classifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from os import walk
from random import shuffle
from sklearn.metrics import classification_report


class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


#create source
sources = {}
trainingPath = './dataset/trainingset/'
for home, dirs, files in walk(trainingPath+'positive'):
    for i,filename in enumerate(files):
        sources[home+'/'+filename] = 'TRAIN_POS_'+str(i)

for home, dirs, files in walk(trainingPath+'negative'):
    for i,filename in enumerate(files):
        sources[home+'/'+filename] = 'TRAIN_NEG_'+str(i)

testPath = './dataset/testset/'
for home, dirs, files in walk(testPath + 'positive'):
    for i, filename in enumerate(files):
        sources[home+'/'+filename] = 'TEST_POS_' + str(i)

for home, dirs, files in walk(testPath + 'negative'):
    for i, filename in enumerate(files):
        sources[home+'/'+filename] = 'TEST_NEG_' + str(i)


sentences = LabeledLineSentence(sources)

model = Doc2Vec(min_count=1, window=15, size=50, sample=1e-4, negative=5, workers=4)

corpus = sentences.to_array()
model.build_vocab(corpus)

for epoch in range(15):
    model.train(sentences.sentences_perm())

train_arrays = numpy.zeros((640, 50))
train_labels = numpy.zeros(640)

test_arrays = numpy.zeros((160, 50))
test_labels = numpy.zeros(160)

train_count=0
test_count=0
tagNames = sources.values()
for tag in tagNames:
    if 'TRAIN' in tag:
        train_arrays[train_count] = model.docvecs[tag]
        if 'POS' in tag:
            train_labels[train_count] = 1
        else:
            train_labels[train_count] = 0
        train_count += 1
    elif 'TEST' in tag:
        test_arrays[test_count] = model.docvecs[tag]
        if 'POS' in tag:
            test_labels[test_count] = 1
        else:
            test_labels[test_count] = 0
        test_count += 1


classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)
pred_labels = classifier.predict(test_arrays)
print 'Logistic:'
print classification_report(test_labels,pred_labels)

classifier = SVC()
classifier.fit(train_arrays, train_labels)
pred_labels = classifier.predict(test_arrays)
print 'SVM:'
print classification_report(test_labels,pred_labels)