import numpy as np
import pandas as pd
import os
import math
import time
from pathlib import Path

from part1 import getData

def validate(kernel, alpha, y, gtY):
    yHats = np.sign(np.dot((alpha * np.transpose(y).flatten()), kernel))
    return np.sum(yHats == np.transpose(gtY).flatten())
    '''
    correct = 0
    y = np.transpose(y).flatten()
    gtY = np.transpose(gtY).flatten()

    for j in range(kernel.shape[1]):
        if (np.sign(np.sum(kernel[:, j] * alpha * y)) == gtY[j]):
            correct += 1

    return correct
    '''

def polynomialKernel(x1, x2, p):
    return (1 + np.dot(x1, np.transpose(x2))) ** p

def onlineKernelPerceptron(x, y, kernel, p, limit=100, outFile=None, validateX=None, validateY=None):
    n = len(x)
    validateN = len(validateX)
    
    alpha = np.zeros(n)
    
    timerStart = time.perf_counter()
    K = kernel(x, x, p) # Training
    timerStop = time.perf_counter()
    print("Gram Matrix Training: ", timerStop - timerStart, "seconds")

    timerStart = time.perf_counter()
    Kv = kernel(x, validateX, p) # Validation
    timerStop = time.perf_counter()
    print("Gram Matrix Validation: ", timerStop - timerStart, "seconds")

    s = 0

    _s = []
    _trainW = []
    _validateW = []
    _runtime = []

    report = pd.DataFrame()
    Path("./part2.a.output/").mkdir(parents=True, exist_ok=True)

    trainAccuracy = 0
    validateAccuracy = 0
    
    timerStart = time.perf_counter()

    while True:

        for idx, row in enumerate(K):
            if (np.sign(np.sum(alpha * row * np.transpose(y).flatten())) != y[idx]):
                alpha[idx] += 1
            print('\r[', int(((idx+1) / n) * 100), "%] S:", s, "P:", p, "Train %:", trainAccuracy, "Validate %:", validateAccuracy, end=' ')
        

        _s.append(s)
        trainAccuracy = (validate(K, alpha, y, y) / n) * 100
        validateAccuracy = (validate(Kv, alpha, y, validateY) / validateN) * 100

        _trainW.append(trainAccuracy)
        _validateW.append(validateAccuracy)
        _runtime.append(time.perf_counter() - timerStart)

        s += 1
        if (s == (limit)):
            break
            
    print('\r[', int(((idx+1) / n) * 100), "%] S:", s, "P:", p, "Train %:", trainAccuracy, "Validate %:", validateAccuracy, end=' ')
    print("\r")
    
    report['iteration'] = _s
    report[str(p) + 'train'] = _trainW
    report[str(p) + 'validate'] = _validateW
    report[str(p) + 'runtime'] = _runtime
    report.to_csv(os.path.join("./part2.a.output/", outFile), index=False)
    return
    
if __name__ == '__main__':
    trainFeatures = getData("./data/pa3_train_X.csv")
    trainGT = getData("./data/pa3_train_y.csv")

    testFeatures = getData("./data/pa3_dev_X.csv")
    testGT = getData("./data/pa3_dev_y.csv")

    p = [1, 2, 3, 4, 5]
    for idx, _p in enumerate(p):
        onlineKernelPerceptron(trainFeatures, trainGT, polynomialKernel, _p, outFile="part2.a.p"+str(_p)+".csv", validateX=testFeatures, validateY=testGT)
