from contentView import Word_Eembedding_Method
from unigram import Unigram
reviewers = {}
with open('./amazon/colluder_labels.txt') as f:
    for line in f:
        items = line.strip().split('\t')
        reviewers[items[0]] = int(items[1])
#
#################################################################
negative = []
positive = []

with open('cut_words.txt') as f:
    for line in f:
        items = line.strip().split('\t')
        if reviewers[items[0]] == 1:
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


we = Word_Eembedding_Method(trainingDict,testDict,reviewers)
we.trainingNet(10,200)
we.prepareData()
label1 = we.fitAndPredict()


########################################################################
#prepare data for DegreeSAD
from loadRating import load
userProfile,itemProfile = load('./amazon/filter_reviews.txt',0,1,3,'\t')
# normalData = {}
# dirtyData = {}
#
# for key in reviewers:
#     if reviewers[key] == '1':
#         dirtyData[key] = userProfile[key]
#
# for key in reviewers:
#     if reviewers[key] == '0' and len(normalData)<len(dirtyData):
#         normalData[key] = userProfile[key]


# for key in ra.userProfile:
#     if len(ra.userProfile[key])>=5 and len(normalData)<len(dirtyData):
#         normalData[key] = ra.userProfile[key]
#prepare data
trainingData = {}
testData = {}
label = {}


import random
for user in trainingDict:
    trainingData[user] = userProfile[user]
    label[user] = int(reviewers[user])

for user in testDict:
    testData[user] = userProfile[user]
    label[user] = int(reviewers[user])

# for key in ra.userProfile:
#     if len(ra.userProfile[key])>=5 and len(normalData)<len(dirtyData):
#         normalData[key] = ra.userProfile[key]
# print 'data picked out...'

from DegreeSAD import DegreeSAD
de = DegreeSAD(trainingData,testData,itemProfile,label)
de.prepareData()
label2 = de.fitAndPredict()


print 'ratingReview Approach'
from ratingView import RatingView
de = RatingView(trainingData,testData,itemProfile,label)
de.prepareData()
label3 = de.fitAndPredict()



groundTruth = []
predict = []
l1 = []
l2 = []
l3 = []
for user in label2:
    l1.append(label1[user][0])
    l2.append(label2[user][0])
    l3.append(label3[user][0])
    groundTruth.append(reviewers[user])
    if label1[user]+label2[user]+label3[user]<2:
        predict.append(0)

    else:
        predict.append(1)

from sklearn.metrics import classification_report

print 'Word-Embedding Approach'
print classification_report(groundTruth,l1)
print 'DegreeSAD Approach'
print classification_report(groundTruth,l2)
print 'ratingReview Approach'
print classification_report(groundTruth,l3)
print 'triViews Approach'
print classification_report(groundTruth,predict)
