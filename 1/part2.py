import numpy as np
import pandas as pd
import os
from pathlib import Path

from part0 import splitDates, normalizeNumerical
from part1 import getFeaturesAndGT, saveReport, getMSE

np.seterr(invalid='ignore')

def batchGradientDescent(learningRate, features, gt, reportName=False):
    w = np.zeros(features.shape[1], dtype=np.float64)
    n = gt.shape[0]
    _mse = []
    for i in range(0, 10000): # Part 2 - "For each value, train up to 10000 iterations ( Fix the number of iterations for this part)"
        gradient = (2 / n) * (np.dot( (np.dot(features, w) - gt), features))    # Calculate gradient
        w = np.subtract(w, np.dot(learningRate, gradient) )                          # Update weight
        theMSE = getMSE(features, gt, w)                                        # Calculate MSE
        norm = np.linalg.norm(gradient)                                         # Calculate norm of gradient
        _mse.append(theMSE)                                                     # record MSE
        print('\r[', i+1, "] LR:", learningRate, "Train MSE:", theMSE, "Norm:", norm,end=' ') # Part 1, b - "For each learning rate that worked for you, Report the MSE on the training data and the validation
                                                                                            # data respectively and the number of iterations needed to achieve the convergence condition for training."

    print("\r")

    if (reportName != False):
        saveReport(learningRate, reportName, _mse, "./part2.Output/")

    return w

learningRates = [100, 10, 0.1, 0.01, 0.001, 0.0001] # Part 2 - "Consider at least the following values for learning rate: 100, 10, 10−1, 10−2, 10−3, 10−4"

print("\n\t>Part 2 - Process un-normalized data for the following learning rates, up to 10000 iterations.")
print("Learning Rates:\n", learningRates, "\n")

trainFeaturesNotNormalized, trainGTNotNormalized = getFeaturesAndGT("./data/PA1_train.csv", normalize=False) # Prep training features not normalized
testFeaturesNotNormalized, testGTNotNormalized = getFeaturesAndGT("./data/PA1_dev.csv", normalize=False)     # Prep testing features not normalized

w = None
for lr in learningRates:
    w = batchGradientDescent(lr, trainFeaturesNotNormalized, trainGTNotNormalized, reportName="PA1_train_Part2NotNorm")
    mse = getMSE(testFeaturesNotNormalized, testGTNotNormalized, w)   # Calculate Test MSE
    print("Test MSE: ", mse)    # Part 1, c - "Use the validation data to pick the best converged solution, and report the learned weights for each feature."
                                # Which features are the most important in deciding the house prices according to the learned
                                # weights? Compare them to your pre-analysis results (Part 0 (d))"
    print(w)
    print("")

print("\n\t>Part 2 - Process normalized data for the following learning rates, up to 10000 iterations.")
print("Learning Rates:\n", learningRates, "\n")

trainFeaturesNormalized, trainGTNormalized = getFeaturesAndGT("./data/PA1_train.csv", normalize=True) # Prep training features normalized
testFeaturesNormalized, testGTNormalized = getFeaturesAndGT("./data/PA1_dev.csv", normalize=True)     # Prep testing features normalized

w = None
for lr in learningRates:
    w = batchGradientDescent(lr, trainFeaturesNormalized, trainGTNormalized, reportName="PA1_train_Part2Norm")
    mse = getMSE(testFeaturesNormalized, testGTNormalized, w)   # Calculate Test MSE
    print("Test MSE: ", mse)    # Part 1, c - "Use the validation data to pick the best converged solution, and report the learned weights for each feature."
                                # Which features are the most important in deciding the house prices according to the learned
                                # weights? Compare them to your pre-analysis results (Part 0 (d))"
    print(w)
    print("")