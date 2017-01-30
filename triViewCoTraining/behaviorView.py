import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from random import shuffle
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from math import log
from collections import defaultdict


class BehaviorView(object):
    def __init__(self, trainingData, testData, itemProfile,brandProfile, label):
        self.trainingDict = trainingData
        self.testDict = testData
        self.labelDict = label
        self.itemProfile = itemProfile
        self.brandProfile = brandProfile

    def prepareData(self):
        self.trainingSet = []
        self.testSet = []
        self.trainingLabel = []
        self.testLabel = []
        self.entropy = {}
        self.FMD = {}
        self.BDS = {}

        # compute the behavior features
        for user in self.trainingDict:
            bds = 0
            brandsCount = defaultdict(int)
            total = len(self.trainingDict[user])
            for item in self.trainingDict[user]:
                if not self.brandProfile.has_key(item):
                    continue
                brandsCount[self.brandProfile[item]]+=1
            for i in brandsCount.values():
                i = i*1.0
                bds+=i/total * log(i/total,2)
            self.BDS[user] = -bds

        for user in self.testDict:
            bds = 0
            brandsCount = defaultdict(int)
            total = len(self.testDict[user])
            for item in self.testDict[user]:
                if not self.brandProfile.has_key(item):
                    continue
                brandsCount[self.brandProfile[item]]+=1
            for i in brandsCount.values():
                i = i*1.0
                bds+=i/total * log(i/total,2)
            self.BDS[user] = -bds


        # prepare the data set
        for user in self.trainingDict:
            self.trainingSet.append([self.BDS[user]])
            self.trainingLabel.append(self.labelDict[user])

    def fitAndPredict(self):
        # classifier = LogisticRegression()
        # classifier.fit(self.trainingSet, self.trainingLabel)
        # pred_labels = classifier.predict(self.testSet)
        # print 'Logistic:'
        # print classification_report(self.testLabel, pred_labels)

        self.classifier = SVC()
        self.classifier.fit(self.trainingSet, self.trainingLabel)
        pred_labels = {}
        for user in self.testDict:
            pred_labels[user] = self.classifier.predict([[self.BDS[user]]])
        # print 'SVM:'
        # print classification_report(self.testLabel, pred_labels)

        # classifier = DecisionTreeClassifier(criterion='entropy')
        # classifier.fit(self.trainingSet, self.trainingLabel)
        # pred_labels = classifier.predict(self.testSet)
        # print 'Decision Tree:'
        # print classification_report(self.testLabel, pred_labels)
        # return self.trainingSet, self.trainingLabel, self.testSet, self.testLabel

        return pred_labels

    def fit(self,trainingSet,trainingLabel):
        self.classifier.fit(trainingSet,trainingLabel)

    def predict(self,sample):
        return self.classifier.predict([])

