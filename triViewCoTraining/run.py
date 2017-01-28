from contentView import Word_Eembedding_Method
from unigram import Unigram
reviewers = {}
with open('./amazon/colluder_labels.txt') as f:
    for line in f:
        items = line.strip().split('\t')
        reviewers[items[0]] = items[1]
#
#################################################################
negative = []
positive = []

with open('cut_words.txt') as f:
    for line in f:
        items = line.strip().split('\t')
        if reviewers[items[0]] == '1':
            negative.append([items[1].split('/'),items[0]])
        else:
            positive.append([items[1].split('/'),items[0]])

balanced_positive = []
import random
while len(balanced_positive)<len(negative):
    index = random.randint(0,len(positive)-1)
    balanced_positive.append(positive[index])
    del positive[index]

count = len(positive)*2
trainSet = []
testSet = []
while len(positive)!=0 and len(negative)!=0:
    if random.random()<0.8:
        if random.random()<0.5:
            trainSet.append(positive[0])
            del positive[0]
        else:
            trainSet.append(negative[0])
            del negative[0]
    else:
        if random.random() < 0.5:
            testSet.append(positive[0])
            del positive[0]
        else:
            testSet.append(negative[0])
            del negative[0]

trainingDict = {item[1]:item[0] for item in trainSet}
testDict = {item[1]:item[0] for item in testSet}

trainingSet_f = [item[0] for item in trainSet]
trainLabels_f = [int(reviewers[item[1]]) for item in trainSet]

testSet_f = [item[0] for item in testSet]
testLabel_f = [int(reviewers[item[1]]) for item in testSet]

# print 'Unigram Approach'
# import unigram
# unigram.PCA_Applied = True
# uni = Unigram(trainingSet_f,testSet_f,trainLabels_f,testLabel_f)
# uni.fitAndPredict()
# print 'TF-IDF Approach'
# import TF_IDF
# # TF_IDF.PCA_Applied = True
# tf = TF_IDF.TF_IDF(trainingSet_f,testSet_f,trainLabels_f,testLabel_f)
# tf.fitAndPredict()
print 'Word-Embedding Approach'
we = Word_Eembedding_Method(trainingDict,testDict,reviewers)
we.trainingNet(10,200)
we.prepareData()
we.fitAndPredict()