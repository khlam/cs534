import numpy as np
import pandas as pd
import os
import math
import json
from pathlib import Path

def validateRow(row, node):
    if (not isinstance(node, int)):
        val = row[node['name']]
        return validateRow(row, node[val])
    return node

def validateData(data, gt, tree):
    correct = 0
    n = data.shape[0]
    for (idx, row) in data.iterrows():
        classification = validateRow(row, tree)
        if (classification == gt[idx]):
            correct += 1
        print('\rValidating [', int(((idx+1) / n) * 100), "%] Pred:", classification, "Actual", gt[idx], "Accuracy:", round((correct / n * 100), 2), end=' ')
    print("\r")
    return (correct / n * 100)

def entropy(col):
    prob = np.bincount(col) / len(col)
    #print(prob, len(col))
    _entropy = 0
    for p in prob:
        if p > 0:
            _entropy += -1 * (p * np.log2(p))
    return _entropy

def leftRight(data, splitIDX, gt):
    leftIDX = np.where(data[:, splitIDX] == 0)
    rightIDX = np.where(data[:, splitIDX] == 1)
    return data[leftIDX], gt[leftIDX], data[rightIDX], gt[rightIDX]

def informationGain(data, splitIDX, gt):
    n = data.shape[0]
    left, leftGT, right, rightGT = leftRight(data, splitIDX, gt)

    eT = entropy(gt)
    eTv = ((left.shape[0] / n) * entropy(leftGT)) + ((right.shape[0] / n) * entropy(rightGT))

    ig = eT - eTv
    return ig, left, leftGT, right, rightGT

def split(data, featureIDX, gt, maxDepth, currentDepth):
    splitIDX = None
    ig = 0
    left = None
    leftGT = None
    right = None
    rightGT = None

    for i, ftIDX in enumerate(featureIDX):
        _ig, _left, _leftGT, _right, _rightGT = informationGain(data, ftIDX, gt)

        if (_ig >= ig):
            splitIDX = ftIDX
            ig, left, leftGT, right, rightGT = _ig, _left, _leftGT, _right, _rightGT

    return splitIDX, ig, left, leftGT, right, rightGT
 
def trainDecisionTree(featuresArray, featureNames, gt, maxDepth, currentDepth=0, LR=None):
    print('\rTraining dMax=', maxDepth,'[', int(((currentDepth) / maxDepth) * 100), "%] ", currentDepth, "/", maxDepth,end=' ')

    if ((currentDepth) == maxDepth):
        return LR

    featureIDX, ig, left, gtLeft, right, gtRight = split(featuresArray, np.arange(0, len(featureNames)), gt, maxDepth, currentDepth)
    
    node = {}
    node['name'] = featureNames[featureIDX]
    node['idx'] = featureIDX
    node['ig'] = ig
    if (left.shape[0] != 0):
        node[0] = trainDecisionTree(left, featureNames, gtLeft, maxDepth, currentDepth+1, int(np.bincount(gtLeft).argmax()))      # left
    else:
        node[0] = int(np.bincount(gtRight).argmin())
    if (right.shape[0] != 0):
        node[1] = trainDecisionTree(right, featureNames, gtRight, maxDepth, currentDepth+1, int(np.bincount(gtRight).argmax()))    # right
    else:
        node[1] = int(np.bincount(gtLeft).argmin())
    return node

def getData(name, _header='infer'):
    df = pd.read_csv(name, header=_header)
    df.apply(pd.to_numeric, errors='ignore')
    sortedColumns = sorted(list(set(df.columns)))
    df = df[sortedColumns]
    return df

if __name__ == '__main__':
    trainFeatDF = getData("./data/pa4_train_X.csv")
    trainFeatArray = trainFeatDF.to_numpy(dtype=int)
    trainGTArray = getData("./data/pa4_train_y.csv", None).to_numpy(dtype=int).flatten()

    testFeatDF = getData("./data/pa4_dev_X.csv")
    testGTArray = getData("./data/pa4_dev_y.csv", None).to_numpy(dtype=int).flatten()

    depths = list((np.arange(1,11)*5).flatten()) 
    depths.append(2)
    depths = sorted(depths) # depths [2, 5, 10, 15 ... 50]

    Path("./part1/").mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame()
    #_trees = []
    _depths = []
    _training = []
    _validation = []

    for _maxDepth in depths:
        print("\n")
        tree = trainDecisionTree(trainFeatArray, list(trainFeatDF.columns), trainGTArray, maxDepth=_maxDepth)
        print("\r")
        _depths.append(_maxDepth)
        #_trees.append(json.dumps(tree))
        _training.append(validateData(trainFeatDF, trainGTArray, tree))
        _validation.append(validateData(testFeatDF, testGTArray, tree))

    report['depth'] = _depths
    report['training'] = _training
    report['validation'] = _validation
    #report['tree'] = _trees
    report.to_csv(os.path.join("./part1/", "part1.csv"), index=False)