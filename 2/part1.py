import numpy as np
import pandas as pd
import os
import math
from pathlib import Path

limit = 10000

def getBinaryResult(result):
    result[np.where(result > 0.5)] = 1
    result[np.where(result < 0.5)] = 0
    return result

# NumPy friendly numerically stable sigmoid
# https://stackoverflow.com/a/62860170
def sigmoid(x):
    return np.piecewise(x, [x > 0], [lambda i: 1 / (1 + np.exp(-i)), lambda i: np.exp(i) / (1 + np.exp(i))])

def gradientDescentLogisticRegression(learningRate, _lambda, features, gt):
    w = np.random.normal(0,1, features.shape[1])
    n = features.shape[0]

    i = 0

    while True:
        w = w + ( (learningRate / n) * np.dot( (np.transpose(gt).flatten() - sigmoid(np.dot(features, w))), features ))
        
        _saveDummy = w[-1]
        w = np.subtract(w, ((learningRate * _lambda) * w))
        w[-1] = _saveDummy
        
        i+=1

        _result = getBinaryResult( sigmoid(np.dot(features, w)) )
        trainCorrect = np.sum(_result == np.transpose(gt).flatten())
        
        print('\r[', i, "] [Part 1; LR:", learningRate, " Reg:", _lambda, "] Training Accuracy: ", trainCorrect, "/", features.shape[0], "(", ((trainCorrect / features.shape[0]) * 100), "%)",end=' ')

        if (i == limit):
            break

    #print("\r")

    return w

def preprocessFeatures(fname):
    # Numerically normalizes a given column
    def normalize(column):
        _max = np.amax(column)
        _min = np.amin(column)
        _denom = _max - _min
        _new = []
        for i, val in enumerate(column):
            _new.append( (val - _min) / (_denom) )
        return _new

    df = pd.read_csv(fname)

    df['Age'] = normalize(df['Age'])
    df['Annual_Premium'] = normalize(df['Annual_Premium'])
    df['Vintage'] = normalize(df['Vintage'])

    return df.to_numpy(dtype=float), df.columns.to_numpy()

def merge(list1, list2): 
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))] 
    return merged_list

if __name__ == '__main__':
    trainFeatures, columnNames = preprocessFeatures("./data/pa2_train_X.csv")
    trainGT = pd.read_csv("./data/pa2_train_y.csv").to_numpy(dtype=float)
    
    validationFeatures, _ = preprocessFeatures("./data/pa2_dev_X.csv")
    validationGT = pd.read_csv("./data/pa2_dev_y.csv").to_numpy(dtype=float)

    regularization = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    learningRate = 0.1

    for idx, regValue in enumerate(regularization):
        w = gradientDescentLogisticRegression(learningRate, regValue, trainFeatures, trainGT)
        _validateResult = getBinaryResult( sigmoid(np.dot(validationFeatures, w)) )
        validateCorrect = np.sum(_validateResult == np.transpose(validationGT).flatten())
        
        print("\nValidation Accuracy: ", validateCorrect, "/", validationFeatures.shape[0], "(", (validateCorrect / validationFeatures.shape[0]) * 100, "%)")
        
        _absW = np.absolute(w)
        _sortedWIndex = np.argsort(_absW)
        
        print(merge(columnNames[_sortedWIndex], _absW[_sortedWIndex]))
        
        print("\n")

    