import numpy as np
import pandas as pd
import os
import math
import copy
from pathlib import Path

limit = 10000

from part1 import sigmoid, getBinaryResult, preprocessFeatures, merge

def gradientDescentLogisticRegression(learningRate, _lambda, features, gt):
    w = np.random.normal(0,1, features.shape[1])
    n = features.shape[0]

    i = 0

    while True:
        w = w + ( (learningRate * (1 / n)) * np.dot( (np.transpose(gt).flatten() - sigmoid(np.dot(features, w))), features ))
        
        _saveDummy = w[-1]
        _sign = np.sign(w)
        _wCopy = np.subtract(np.absolute(w), (learningRate * _lambda))
        _wCopy[np.where(_wCopy < 0)] = 0
        w = _sign * _wCopy
        w[-1] = _saveDummy

        i+=1

        _result = getBinaryResult( sigmoid(np.dot(features, w)) )
        trainCorrect = np.sum(_result == np.transpose(gt).flatten())
        
        print('\r[', i, "] [Part 2; LR:", learningRate, " Reg:", _lambda, "] Training Accuracy: ", trainCorrect, "/", features.shape[0], "(", ((trainCorrect / features.shape[0]) * 100), "%)",end=' ')

        if (i == limit):
            break

    print("\r")

    return w

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

        print("0 Count:", len(np.where(w == 0)[0]))
        
        print("\nValidation Accuracy: ", validateCorrect, "/", validationFeatures.shape[0], "(", (validateCorrect / validationFeatures.shape[0]) * 100, "%)")
        
        _absW = np.absolute(w)
        _sortedWIndex = np.argsort(_absW)
        
        print(merge(columnNames[_sortedWIndex], _absW[_sortedWIndex]))
        
        print("\n")
