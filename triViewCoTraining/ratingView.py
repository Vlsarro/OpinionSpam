import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from random import shuffle
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from math import log
class RatingView(object):
    def __init__(self,trainingData,testData,itemProfile,label):
        self.trainingDict = trainingData
        self.testDict = testData
        self.labelDict = label
        self.itemProfile = itemProfile

    def prepareData(self):
        self.trainingSet = []
        self.testSet = []
        self.trainingLabel = []
        self.testLabel = []
        self.MUD = {}
        self.RUD = {}
        self.QUD = {}
        self.entropy = {}
        self.FMD = {}
        #compute the rating features
        sList = sorted(self.itemProfile.iteritems(),key=lambda d:len(d[1]),reverse=True)
        maxLength = len(sList[0][1])

        for user in self.trainingDict:
            self.MUD[user] = 0
            for item in self.trainingDict[user]:
                self.MUD[user]+= len(self.itemProfile[item])/float(maxLength)
        for user in self.trainingDict:
            lengthList = [len(self.itemProfile[item]) for item in self.trainingDict[user]]
            lengthList.sort(reverse=True)
            self.RUD[user]=lengthList[0]-lengthList[-1]
        for user in self.trainingDict:
            lengthList = [len(self.itemProfile[item]) for item in self.trainingDict[user]]
            lengthList.sort()
            self.QUD[user] = lengthList[int((len(lengthList)-1)/4.0)]
        for user in self.trainingDict:
            entropy = 0
            total = sum(self.trainingDict[user].values())*1.0
            for item in self.trainingDict[user]:
                rating = self.trainingDict[user][item]
                entropy += rating/total * log(rating/total,2)
            self.entropy[user] = -entropy
        for user in self.trainingDict:
            fmd = 0
            rMean = sum(self.trainingDict[user].values())*1.0/len(self.trainingDict[user])
            for item in self.trainingDict[user]:
                fmd += abs(self.trainingDict[user][item] - rMean)
            self.FMD[user] = fmd/len(self.trainingDict[user])
        


        for user in self.testDict:
            self.MUD[user] = 0
            for item in self.testDict[user]:
                self.MUD[user] += len(self.itemProfile[item]) / float(maxLength)
        for user in self.testDict:
            lengthList = [len(self.itemProfile[item]) for item in self.testDict[user]]
            lengthList.sort(reverse=True)
            self.RUD[user] = lengthList[0] - lengthList[-1]
        for user in self.testDict:
            lengthList = [len(self.itemProfile[item]) for item in self.testDict[user]]
            lengthList.sort()
            self.QUD[user] = lengthList[int((len(lengthList) - 1) / 4.0)]
        for user in self.testDict:
            entropy = 0
            total = sum(self.testDict[user].values())*1.0
            for item in self.testDict[user]:
                rating = self.testDict[user][item]
                entropy += rating/total * log(rating/total,2)
            self.entropy[user] = -entropy
        for user in self.testDict:
            fmd = 0
            rMean = sum(self.testDict[user].values())*1.0/len(self.testDict[user])
            for item in self.testDict[user]:
                fmd += abs(self.testDict[user][item] - rMean)
            self.FMD[user] = fmd/len(self.testDict[user])


        #prepare the data set
        for user in self.trainingDict:
            self.trainingSet.append([self.entropy[user],self.FMD[user]])
            self.trainingLabel.append(self.labelDict[user])



    def fitAndPredict(self):
        # classifier = LogisticRegression()
        # classifier.fit(self.trainingSet, self.trainingLabel)
        # pred_labels = classifier.predict(self.testSet)
        # print 'Logistic:'
        # print classification_report(self.testLabel, pred_labels)

        classifier = SVC()
        classifier.fit(self.trainingSet, self.trainingLabel)
        pred_labels = {}
        for user in self.testDict:
            pred_labels[user] = classifier.predict([[self.entropy[user], self.FMD[user]]])
        # print 'SVM:'
        #print classification_report(self.testLabel, pred_labels)

        # classifier = DecisionTreeClassifier(criterion='entropy')
        # classifier.fit(self.trainingSet, self.trainingLabel)
        # pred_labels = classifier.predict(self.testSet)
        # print 'Decision Tree:'
        # print classification_report(self.testLabel, pred_labels)
        # return self.trainingSet, self.trainingLabel, self.testSet, self.testLabel

        return pred_labels
