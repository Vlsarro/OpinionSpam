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


    def to_vector(self):
        self.sentences = []
        for source in self.sources:
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    words = utils.to_unicode(line).split()
                    words = [word for word in words if word not in stoplist]
                    self.sentences.append(words)
        return self.sentences


#create source
sources = []
labels = []

trainingPath = './dataset/trainingset/'
for home, dirs, files in walk(trainingPath+'positive'):
    for filename in files:
        sources.append(home+'/'+filename)
        labels.append(1)

for home, dirs, files in walk(trainingPath+'negative'):
    for filename in files:
        sources.append(home + '/' + filename)
        labels.append(0)

testPath = './dataset/testset/'
for home, dirs, files in walk(testPath + 'positive'):
    for filename in files:
        sources.append(home + '/' + filename)
        labels.append(1)

for home, dirs, files in walk(testPath + 'negative'):
    for filename in files:
        sources.append(home + '/' + filename)
        labels.append(0)


corpus = Texts(sources).to_vector()
dictionary = corpora.Dictionary(corpus)
corpus = [dictionary.doc2bow(text) for text in corpus]
text_matrix = gensim.matutils.corpus2dense(corpus,num_terms = len(dictionary.token2id)).T
# pca = PCA(n_components=50)
# text_matrix = pca.fit_transform(text_matrix)

classifier = LogisticRegression()
classifier.fit(text_matrix[:640], labels[:640])
pred_labels = classifier.predict(text_matrix[640:])
print 'Logistic:'
print classification_report(labels[640:],pred_labels)

classifier = SVC()
classifier.fit(text_matrix[:640], labels[:640])
pred_labels = classifier.predict(text_matrix[640:])
print 'SVM:'
print classification_report(labels[640:],pred_labels)