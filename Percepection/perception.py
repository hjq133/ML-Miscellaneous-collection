import numpy as np
import time
from tqdm import tqdm


def loadData(fileName):
    '''
    加载Mnist数据集
    :param fileName:要加载的数据集路径
    :return: list形式的数据集及标记
    '''
    print('start to read data')
    # 存放数据及标记的list
    dataArr = []
    labelArr = []
    # 打开文件
    fr = open(fileName, 'r')
    # 将文件按行读取
    for i, line in enumerate(fr.readlines()):
        # 对每一行数据按切割福','进行切割，返回字段列表
        curLine = line.strip().split(',')
        # Mnsit有0-9是个标记，由于是二分类任务，所以将>=5的作为1，<5为-1
        if i == 0:
            continue
        if int(curLine[0]) >= 5:
            labelArr.append(1)
        else:
            labelArr.append(-1)
        # 存放标记
        # [int(num) for num in curLine[1:]] -> 遍历每一行中除了以第一哥元素（标记）外将所有元素转换成int类型
        # [int(num)/255 for num in curLine[1:]] -> 将所有数据除255归一化(非必须步骤，可以不归一化)
        dataArr.append([int(num) / 255 for num in curLine[1:]])
    # 返回data和label
    return dataArr, labelArr


def perception(dataArr, labelArr, iter=50):
    '''
    dataArr : (m, n)  m为样本个数，n为特征数
    labelArr : (m, )
    iter : 迭代次数，因为不大可能达到100%
    '''
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr)
    m, n = np.shape(dataMat)
    w = np.zeros((1, n))
    b = 0
    h = 0.0001

    for k in tqdm(range(iter)):
        for i in range(m):  # 随机梯度下降，遍历每个样本点
            xi = dataMat[i]  # shape (1, 784)
            yi = labelMat[0, i]

            if -1 * yi * (w * xi.T + b) >= 0:
                w = w + h * yi * xi
                b = b + h * yi
    return w, b


def perception(dataArr, labelArr, iter=50):
    '''
    dataArr : (m, n)  m为样本个数，n为特征数
    labelArr : (m, )
    iter : 迭代次数，因为不大可能达到100%
    '''
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr)
    m, n = np.shape(dataMat)
    w = np.zeros((1, n))
    b = 0
    h = 0.0001

    for k in tqdm(range(iter)):
        for i in range(m):  # 随机梯度下降，遍历每个样本点
            xi = dataMat[i]  # shape (1, 784)
            yi = labelMat[0, i]

            if -1 * yi * (w * xi.T + b) >= 0:
                w = w + h * yi * xi
                b = b + h * yi
    return w, b


def test(dataArr, labelArr, w, b):
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr)
    m, n = np.shape(dataMat)
    errCnt = 0

    for i in tqdm(range(m)):
        xi = dataMat[i]
        yi = labelMat[0, i]

        if -1 * yi * (w * xi.T + b) >= 0:  # 错误分类的
            errCnt += 1

    accuracy = float(m - errCnt) / float(m)
    return accuracy


trainData, trainLabel = loadData('../../data/Mnist/train.csv')
testData, testLabel = loadData('../../data/Mnist/test.csv')

w, b = perception(trainData, trainLabel, 30)

acc = test(trainData, trainLabel, w, b)

print(acc)

