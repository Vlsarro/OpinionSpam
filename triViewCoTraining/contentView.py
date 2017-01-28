# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
# numpy
import numpy
# classifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from random import shuffle
from sklearn.metrics import classification_report
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier


class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if key not in flipped:
                flipped[key] = [value]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            yield LabeledSentence(source, [prefix])

    def to_array(self):
        self.sentences = []
        for prefix, source in self.sources.items():
            self.sentences.append(LabeledSentence(source, [prefix]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


class Word_Eembedding_Method(object):
    def __init__(self, trainingData, testData, label):
        self.corpus = {}
        self.trainingDict = trainingData
        self.testDict = testData
        self.labelDict = label

        for user in trainingData:
            tag = user
            self.corpus[tag] = trainingData[user]

        for user in testData:
            tag = user
            self.corpus[tag] = testData[user]


    def trainingNet(self, window, nDimension):
        self.nDimension = nDimension
        sentences = LabeledLineSentence(self.corpus)
        self.model = Doc2Vec(min_count=1, window=window, size=nDimension, sample=1e-4, negative=5, workers=4)
        corpus = sentences.to_array()
        self.model.build_vocab(corpus)
        for epoch in range(10):
            self.model.train(sentences.sentences_perm())

    def prepareData(self):
        self.trainingSet = []
        self.testSet = []
        self.trainingLabel = []
        self.testLabel = []

        for i, user in enumerate(self.trainingDict):
            self.trainingSet.append(self.model.docvecs[user])
            self.trainingLabel.append(self.labelDict[user])

        for i, user in enumerate(self.testDict):
            self.testSet.append(self.model.docvecs[user])
            self.testLabel.append(self.labelDict[user])

        return self.trainingSet, self.trainingLabel, self.testSet, self.testLabel

    def fitAndPredict(self):
        classifier = LogisticRegression()
        classifier.fit(self.trainingSet, self.trainingLabel)
        pred_labels = classifier.predict(self.testSet)
        print 'Logistic:'
        print classification_report(self.testLabel, pred_labels)

        classifier = SVC()
        classifier.fit(self.trainingSet, self.trainingLabel)
        pred_labels = classifier.predict(self.testSet)
        print 'SVM:'
        print classification_report(self.testLabel, pred_labels)

        classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                                max_depth=1, random_state=0)
        classifier.fit(self.trainingSet, self.trainingLabel)
        pred_labels = classifier.predict(self.testSet)
        print 'GBDT:'
        print classification_report(self.testLabel, pred_labels)

        clf = AdaBoostClassifier(n_estimators=100)
        classifier.fit(self.trainingSet, self.trainingLabel)
        pred_labels = classifier.predict(self.testSet)
        print 'AdaBoost:'
        print classification_report(self.testLabel, pred_labels)

        clf = RandomForestClassifier(n_estimators=10)
        classifier.fit(self.trainingSet, self.trainingLabel)
        pred_labels = classifier.predict(self.testSet)
        print 'Random Forest:'
        print classification_report(self.testLabel, pred_labels)



