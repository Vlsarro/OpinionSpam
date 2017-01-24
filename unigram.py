# gensim modules
import gensim
from gensim import utils
from gensim import corpora
# numpy
import numpy
# classifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from os import walk
from random import shuffle
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

stoplist = set('for a of the and to in'.split())

class Texts(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def to_vector(self):
        self.sentences = []
        self.tags = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    words = utils.to_unicode(line).split()
                    words = [word for word in words if word not in stoplist]
                    self.sentences.append(words)
                    self.tags.append(prefix)
        return self.sentences,self.tags


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


corpus,tagNames = Texts(sources).to_vector()
dictionary = corpora.Dictionary(corpus)
corpus = [dictionary.doc2bow(text) for text in corpus]
text_matrix = gensim.matutils.corpus2dense(corpus,num_terms = len(dictionary.token2id)).T
# pca = PCA(n_components=50)
# text_matrix = pca.fit_transform(text_matrix)

train_arrays = numpy.zeros((640, len(dictionary.token2id)))
train_labels = numpy.zeros(640)

test_arrays = numpy.zeros((160, len(dictionary.token2id)))
test_labels = numpy.zeros(160)

train_count=0
test_count=0

for i,tag in enumerate(tagNames):
    if 'TRAIN' in tag:
        train_arrays[train_count] = text_matrix[i]
        if 'POS' in tag:
            train_labels[train_count] = 1
        else:
            train_labels[train_count] = 0
        train_count += 1
    elif 'TEST' in tag:
        test_arrays[test_count] = text_matrix[i]
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