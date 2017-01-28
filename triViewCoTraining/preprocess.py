#coding:utf8
from collections import defaultdict
import jieba
reviews = defaultdict(list)
stopwords = ['了','的','吗','啊','嘛','吧',' ','。','，','.',',','！','？']
stopwords = [item.decode('utf8') for item in stopwords]
with open('./amazon/filter_reviews.txt') as f:
    for line in f:
        items = line.strip().split('\t')
        content = jieba.cut(items[6].strip().decode('gbk'))
        content = '/'.join([word for word in content if word not in stopwords]).encode('utf8')
        reviews[items[0]].append(content)
with open('cut_words.txt','w') as f:
    for key in reviews:
        f.write(key+'\t'+'/'.join(reviews[key])+'\n')
###################################################################
reviewers = {}
with open('./amazon/colluder_labels.txt') as f:
    for line in f:
        items = line.strip().split('\t')
        reviewers[items[0]] = items[1]
#content = []
# with open('cut_words.txt') as f:
#     for line in f:
#         user,review = line.strip().split('\t')
#         content.append(user+'\t'+review+'\t'+reviewers[user]+'\n')
# with open('labeled_reviews.txt','w') as f:
#     f.writelines(content)
##################################################################
# negative = []
# positive = []
#
# with open('cut_words') as f:
#     for line in f:
#         items = line.strip().split('\t')
#         if reviewers[items[0]] == '1':
#             negative.append([[items[1]],items[0]])
#         else:
#             positive.append([[items[1]],items[0]])
#
# balanced_positive = []
# import random
# while len(balanced_positive)<len(negative):
#     index = random.randint(0,len(positive)-1)
#     balanced_positive.append(positive[index])
#     del positive[index]
#
# count = len(positive)*2
# trainSet = []
# testSet = []
# while len(positive)!=0 and len(negative)!=0:
#     if random.random()<0.8:
#         if random.random()<0.5:
#             trainSet.append(positive[0])
#             del positive[0]
#         else:
#             trainSet.append(negative[0])
#             del negative[0]
#     else:
#         if random.random() < 0.5:
#             testSet.append(positive[0])
#             del positive[0]
#         else:
#             testSet.append(negative[0])
#             del negative[0]
#
# trainingDict = {item[1]:item[0] for item in trainSet}
# testDict = {item[1]:item[0] for item in testSet}

# with open('trainingReview.txt','w') as f:
#     f.writelines(trainSet)
# with open('testReview.txt','w') as f:
#     f.writelines(testSet)