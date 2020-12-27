import numpy as np
import pandas as pd
import os
import math
import json
from pathlib import Path

from part1 import split, validateRow

def forestPredict(row, forest):
    vote = []
    for tree in forest:
        vote.append(validateRow(row, tree))
    return int(np.bincount(vote).argmax())

def validateData(data, gt, forest):
    correct = 0
    n = data.shape[0]
    for (idx, row) in data.iterrows():
        classification = forestPredict(row, forest)
        if (classification == gt[idx]):
            correct += 1
        print('\rValidating [', int(((idx+1) / n) * 100), "%] Pred:", classification, "Actual", gt[idx], "Accuracy:", round((correct / n * 100), 2), end=' ')
    print("\r")
    return (correct / n * 100)

def trainDecisionTree(featuresArray, featureNames, m, gt, maxDepth, currentDepth=0, LR=None):
    print('\rTraining dMax:', maxDepth, 'm:', m,'[', int(((currentDepth) / maxDepth) * 100), "%] ", currentDepth, "/", maxDepth,end=' ')

    if ((currentDepth) == maxDepth):
        return LR

    ftIDX = np.random.choice(np.arange(0, len(featureNames)), size=m, replace=False) # Sample feature names no replacement
    featureIDX, ig, left, gtLeft, right, gtRight = split(featuresArray, ftIDX, gt, maxDepth, currentDepth)

    node = {}
    node['name'] = featureNames[featureIDX]
    node['idx'] = featureIDX
    node['ig'] = ig

    if (left.shape[0] != 0):
        node[0] = trainDecisionTree(left, featureNames, m, gtLeft, maxDepth, currentDepth+1, int(np.bincount(gtLeft).argmax()))      # left
    else:
        node[0] = int(np.bincount(gtRight).argmin())
    if (right.shape[0] != 0):
        node[1] = trainDecisionTree(right, featureNames, m, gtRight, maxDepth, currentDepth+1, int(np.bincount(gtRight).argmax()))    # right
    else:
        node[1] = int(np.bincount(gtLeft).argmin())
    return node


def randomForest(featuresArray, featureNames, gt, treeSize, m, dmax):
    forest = []
    n = featuresArray.shape[0]

    for idx in range(0, treeSize):
        bootstrapIDX = np.random.choice(np.arange(0, n), size=n, replace=True) # Sample n samples with replacement per tree
        bootstrapFeatures = featuresArray[bootstrapIDX]
        bootstrapGT = gt[bootstrapIDX]

        forest.append( trainDecisionTree(bootstrapFeatures, featureNames, m, bootstrapGT, maxDepth=dmax) )

    return forest

def getData(name, _header='infer'):
    df = pd.read_csv(name, header=_header)
    df.apply(pd.to_numeric, errors='ignore')
    #sortedColumns = sorted(list(set(df.columns)))
    #df = df[sortedColumns]
    return df

if __name__ == '__main__':
    trainFeatDF = getData("./data/pa4_train_X.csv")
    trainFeatArray = trainFeatDF.to_numpy(dtype=int)
    trainGTArray = getData("./data/pa4_train_y.csv", None).to_numpy(dtype=int).flatten()

    testFeatDF = getData("./data/pa4_dev_X.csv")
    testGTArray = getData("./data/pa4_dev_y.csv", None).to_numpy(dtype=int).flatten()

    depths = [2, 10, 25]
    
    m = [5, 25, 50, 100]

    trees = list((np.arange(1,11)*10).flatten()) 

    Path("./part2/").mkdir(parents=True, exist_ok=True)

    np.random.seed(1)

    for _maxDepth in depths:
        report = pd.DataFrame()
        _subSampleSize = []
        _training = []
        _validation = []
        _treeSizes = []
        for _m in m:
            for _tSize in trees:
                print("\n")
                print("Tree Size:", _tSize)
                forest = randomForest(trainFeatArray, np.array(trainFeatDF.columns), trainGTArray, treeSize=_tSize, m=_m, dmax=_maxDepth)
                print("\r")
                _subSampleSize.append(_m)
                _training.append(validateData(trainFeatDF, trainGTArray, forest))
                _validation.append(validateData(testFeatDF, testGTArray, forest))
                _treeSizes.append(_tSize)

        report['m'] = _subSampleSize
        report['treeSize'] = _treeSizes
        report['training'] = _training
        report['validation'] = _validation
        report.to_csv(os.path.join("./part2/", "part2.dMax"+str(_maxDepth)+".csv"), index=False)

