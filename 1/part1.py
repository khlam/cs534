import numpy as np
import pandas as pd
import os
from pathlib import Path

from part0 import splitDates, normalizeNumerical

# Part 1, a - "Which learning rate or learning rates did you observe to be good for this particular dataset? What
# learning rates (if any) make gradient descent diverge? Report your observations together with some
# example curves showing the training MSE as a function of training iterations and its convergence or
# non-convergence behaviors."
def saveReport(learningRate, name, _mse, parentFolder):
    report = pd.DataFrame()
    report['iteration'] = np.arange(0, len(_mse))
    report['mse'] = _mse

    Path(parentFolder).mkdir(parents=True, exist_ok=True) # if dir doesn't exist create it

    outFile = os.path.join(parentFolder, str(learningRate) + "." + name + ".csv")
    report.to_csv(outFile, index=False)

def getMSE(features, gt, w): # Calculate MSE
    return (1 / gt.shape[0]) * np.dot( np.transpose( (np.dot(features, w) - gt) ), (np.dot(features, w) - gt))

def batchGradientDescent(learningRate, features, gt, report=False):
    w = np.zeros(features.shape[1], dtype=float)
    n = gt.shape[0]
    _mse = []
    i = 0
    while True:
        gradient = (2 / n) * (np.dot( (np.dot(features, w) - gt), features))    # Calculate gradient
        w = np.subtract(w, np.dot(learningRate, gradient) )                     # Update weight
        theMSE = getMSE(features, gt, w)                                        # Calculate MSE
        norm = np.linalg.norm(gradient)                                         # Calculate norm of gradient
        _mse.append(theMSE)                                                     # record MSE
        print('\r[', i, "] LR:", learningRate, "Train MSE:", theMSE, "Norm:", norm,end=' ') # Part 1, b - "For each learning rate that worked for you, Report the MSE on the training data and the validation
                                                                                            # data respectively and the number of iterations needed to achieve the convergence condition for training."
        i+=1
        if (norm < 0.5): # stop when norm is less than 0.5
            break

    print("\r")

    if (report != False):
        saveReport(learningRate, report, _mse, "./part1.Output/")

    return w

def getFeaturesAndGT(csvFname, normalize=True):
    df = pd.read_csv(csvFname)          # Fetches train data
    df = df.drop('id', 1)               # Part 0, a - "Remove ID Feature"
    df = splitDates(df)                 # Part 0, b - "Split the date feature into three separate numerical features: month, day , and year."
    
    # Normalize (if set to true)
    if (normalize):                     # Part 0, e - "Normalize all numerical features (excluding the housing prices y) to the range 0 and 1 using the training data."
        df = normalizeNumerical(df)

    gt = df['price']                    # price we're trying to predict (ground truth)
    df = df.drop('price', 1)
    features = df.to_numpy(dtype=float)
    print("Fetched Features and GT of", csvFname, "\n\tNormalized: ", normalize, "\n\tFeatures:",features.shape, "\n\tGT:", gt.shape)
    return features, gt

if __name__ == '__main__':
    featuresTrain, gtTrain = getFeaturesAndGT("./data/PA1_train.csv", normalize=True) # Prep training features normalized
    featuresTest, gtTest = getFeaturesAndGT("./data/PA1_dev.csv", normalize=True) # Prep testing features normalized

    learningRates = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001] # Part 1 - "For this part, you will work with the preprocessed and normalized data, 
                                                                                # and consider at least the following values for the learning rate: 100, 10−1, 10−2, 10−3, 10−4, 10−5, 10−6, 10−7."

    print("\n\t>Part 1, c - Run the following learning rates until norm of gradient is less than 0.5. Report weights and MSE of training and Testing data")
    print("Learning Rates:\n", learningRates, "\n")

    w = None
    for lr in learningRates:
        w = batchGradientDescent(lr, featuresTrain, gtTrain, report=("Part1.LR."+str(float(lr))))
        mse = getMSE(featuresTest, gtTest, w) # Calculate MSE
        print("Test MSE: ", mse)    # Part 1, c - "Use the validation data to pick the best converged solution, and report the learned weights for each feature."
                                    # Which features are the most important in deciding the house prices according to the learned
                                    # weights? Compare them to your pre-analysis results (Part 0 (d))"
        print(w)
        print("")
    #print(w)
    #print( np.dot(np.transpose(w), features[3]) , gt[3])