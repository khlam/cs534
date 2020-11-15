import numpy as np
import pandas as pd
import os
import math
import time
from pathlib import Path

from part1 import getData
from part2a import validate, polynomialKernel

def batchKernelPerceptron(x, y, kernel, p, learningRate=1, limit=100, outFile=None, validateX=None, validateY=None):
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
    Path("./part2.b.output/").mkdir(parents=True, exist_ok=True)

    trainAccuracy = 0
    validateAccuracy = 0

    while True:
        timerStart = time.perf_counter()

        u = np.sign(np.dot(K, np.transpose(alpha * np.transpose(y).flatten())))
        alpha[np.where((u * np.transpose(y).flatten()) <= 0)] += 1
        # alpha[np.where((u * np.transpose(y).flatten()) <= 0)] += 1 + learningRate # Learning rate does nothing
        print('\r[', s, "/", limit, "]", "P:", p, "Train %:", trainAccuracy, "Validate %:", validateAccuracy, end=' ')
        
        timerStop = time.perf_counter()

        _s.append(s)
        trainAccuracy = (validate(K, alpha, y, y) / n) * 100
        validateAccuracy = (validate(Kv, alpha, y, validateY) / validateN) * 100

        _trainW.append(trainAccuracy)
        _validateW.append(validateAccuracy)
        _runtime.append(timerStop - timerStart)

        s += 1
        if (s == (limit)):
            break

    print('\r[', s, "/", limit, "]", "P:", p, "Train %:", trainAccuracy, "Validate %:", validateAccuracy, end=' ')
    print("\r")
    
    report['iteration'] = _s
    report[str(p) + 'train'] = _trainW
    report[str(p) + 'validate'] = _validateW
    report[str(p) + 'runtime'] = _runtime
    report.to_csv(os.path.join("./part2.b.output/", outFile), index=False)
    return
    
if __name__ == '__main__':
    trainFeatures = getData("./data/pa3_train_X.csv")
    trainGT = getData("./data/pa3_train_y.csv")

    testFeatures = getData("./data/pa3_dev_X.csv")
    testGT = getData("./data/pa3_dev_y.csv")

    batchKernelPerceptron(trainFeatures, trainGT, polynomialKernel, 2, limit=137, outFile="part2.b.csv", validateX=testFeatures, validateY=testGT)
