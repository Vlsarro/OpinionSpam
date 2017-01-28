# gensim modules
import gensim
from gensim import utils
from gensim import corpora,models
# numpy
import numpy
# classifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from os import walk
from random import shuffle
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

PCA_Applied = False
PCA_nComponents = 100


class TF_IDF(object):
    def __init__(self,trainSet,testSet,trainLabels,testLabels):
        self.trainingSet = trainSet
        self.testSet = testSet
        self.trainingLabel = trainLabels
        self.testLabel = testLabels

    def fitAndPredict(self):
        corpus = self.trainingSet+self.testSet
        dictionary = corpora.Dictionary(corpus)
        corpus = [dictionary.doc2bow(text) for text in corpus]
        model = models.TfidfModel(corpus)
        corpus = [text for text in model[corpus]]
        text_matrix = gensim.matutils.corpus2dense(corpus, num_terms=len(dictionary.token2id)).T

        if PCA_Applied:
            pca = PCA(n_components=PCA_nComponents)
            text_matrix = pca.fit_transform(text_matrix)

        classifier = LogisticRegression()
        classifier.fit(text_matrix[0:len(self.trainingSet)], self.trainingLabel)
        pred_labels = classifier.predict(text_matrix[len(self.trainingSet):])
        print 'Logistic:'
        print classification_report(self.testLabel, pred_labels)

        classifier = SVC()
        classifier.fit(text_matrix[0:len(self.trainingSet)], self.trainingLabel)
        pred_labels = classifier.predict(text_matrix[len(self.trainingSet):])
        print 'SVM:'
        print classification_report(self.testLabel, pred_labels)

