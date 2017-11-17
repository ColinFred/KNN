#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
from os import listdir


def img2vector(filename):
    """
    将图片转化为向量形式。即将二维数组变为一维数组
    """
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    """
    数据预处理
    :return: 属性和标签
    """
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  # 加载训练集
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))  # 用来存放训练集各个向量
    for i in range(m):
        fileNameStr = trainingFileList[i]  # 文件名称
        fileStr = fileNameStr.split('.')[0]  # 将 .txt 字符串剔除
        classNumStr = int(fileStr.split('_')[0])  # 提取出这个文件所代表的数字
        hwLabels.append(classNumStr)  # 将数字放入标签集合中
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)  # 将向量形式的数字文件存入trainingMat中

    return trainingMat, hwLabels  # 返回值是训练集的属性和标签


from sklearn.model_selection import train_test_split  # 随机划分训练集和测试集

X, y = handwritingClassTest()
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("training dataset: {0}".format(x_train.shape[0]))
print("testing dataset: {0}".format(x_test.shape[0]))
print("each sample has {0} features".format(x_train.shape[1]))

from sklearn.neighbors import KNeighborsClassifier  # 引入k近邻算法

estimator = KNeighborsClassifier()  # 目前都使用默认参数
estimator.fit(x_train, y_train)  # 开始训练
y_predicted = estimator.predict(x_test)  # 开始预测
accuracy = np.mean(y_test == y_predicted) * 100  # 准确率
print("the accuracy is {0:.1f}%".format(accuracy))

# 为了避免一次性测试的运气问题，我们引入交叉检验
from sklearn.model_selection import cross_val_score

scores = cross_val_score(estimator, X, y, scoring='accuracy')
average_accuacy = np.mean(scores) * 100
print("the average accuracy is {0:.1f}%".format(average_accuacy))  # 82.3%

avg_scores = []
all_scores = []
parameter_values = list(range(1, 21))
for n_neighbors in parameter_values:
    estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(estimator, X, y, scoring='accuracy')
    avg_scores.append(np.mean(scores))
    all_scores.append(scores)

# 使用图来分析一下n_neighbors的不同取值与分类正确率的关系
from matplotlib import pyplot as plt

plt.figure(figsize=(32, 20))
plt.plot(parameter_values, avg_scores, '-o', linewidth=5, markersize=24)
plt.show()
