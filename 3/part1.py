import numpy as np
import pandas as pd
import os
import math
from pathlib import Path

def validate(x, y, w):
    return np.sum(np.sign(np.dot(x, w)) == np.transpose(y).flatten())

def perceptron(x, y, outFile=None, validateX=None, validateY=None, limit=100):
    n = len(x)
    nV = None

    _s = []
    _trainW = []
    _validateW = []
    _trainAvgW = []
    _validateAvgW = []

    if (isinstance(validateX, np.ndarray)):
        nV = len(validateX)
        report = pd.DataFrame()
        Path("./part1.output/").mkdir(parents=True, exist_ok=True)

    w = np.zeros(x.shape[1])
    avgW = np.zeros(x.shape[1])
    
    itr = 1 # Training loop iterator

    s = 1 # example counter
    while True:
        for i, xi in enumerate(x):
            _class = y[i] * np.dot(np.transpose(w), xi)
            if (_class <= 0):
                w = w + (y[i] * xi)
            
            avgW = ((s*avgW) + w) / (s + 1)
            s += 1
            print('\r[', itr, "]", s ,"/", n,end=' ')
    
        if (nV):
            _s.append(s)
            _trainW.append((validate(x, y, w) / n) * 100)
            _validateW.append((validate(validateX, validateY, w) / nV) * 100)
            _trainAvgW.append((validate(x, y, avgW) / n) * 100)
            _validateAvgW.append((validate(validateX, validateY, avgW) / nV) * 100)

        itr += 1
        if (itr == (limit+1)):
            break
    print("\r")   
    if (nV):
        report['iteration'] = _s
        report['OnlinePerceptronTrain'] = _trainW
        report['OnlinePerceptronValidate'] = _validateW
        report['AvgPerceptronTrain'] = _trainAvgW
        report['AvgPerceptronValidate'] = _validateAvgW
        report.to_csv(os.path.join("./part1.output/", outFile), index=False)

    return w, avgW

def getData(fname):
    df = pd.read_csv(fname)
    return df.to_numpy(dtype=float)

if __name__ == '__main__':
    trainFeatures = getData("./data/pa3_train_X.csv")
    trainGT = getData("./data/pa3_train_y.csv")

    testFeatures = getData("./data/pa3_dev_X.csv")
    testGT = getData("./data/pa3_dev_y.csv")

    _, _ = perceptron(trainFeatures, trainGT, "part1.a.csv", testFeatures, testGT) # Part 1 a

    # Part 1 c
    _, avgW = perceptron(trainFeatures, trainGT, limit=200)
    print("Avg Perceptron validate accuracy (%): ", validate(testFeatures, testGT, avgW) / len(testGT) * 100)
    _, avgW = perceptron(trainFeatures, trainGT, limit=500)
    print("Avg Perceptron validate accuracy (%): ", validate(testFeatures, testGT, avgW) / len(testGT) * 100)
    _, avgW = perceptron(trainFeatures, trainGT, limit=1000)
    print("Avg Perceptron validate accuracy (%): ", validate(testFeatures, testGT, avgW) / len(testGT) * 100)
    _, avgW = perceptron(trainFeatures, trainGT, limit=2000)
    print("Avg Perceptron validate accuracy (%): ", validate(testFeatures, testGT, avgW) / len(testGT) * 100)
