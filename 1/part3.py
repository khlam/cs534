import numpy as np
import pandas as pd
import os
from pathlib import Path
import copy
import time
from datetime import date
import random

from part0 import splitDates, normalize
from part1 import saveReport

def predict(w, feature):
    return np.exp(np.dot(np.transpose(w), feature))

def getMSE(features, gt, w): # Calculate MSE
    return (1 / gt.shape[0]) *  np.dot( np.transpose( np.exp( (np.dot(features, w) - gt)) ), np.exp((np.dot(features, w) - gt)) )
    #return (1 / gt.shape[0]) * np.dot( np.transpose( (np.dot(features, w) - gt) ), (np.dot(features, w) - gt))

def equivalence(df, featureName, xName, x):
    _above = str(featureName) + "_IsAbove_" + str(xName)
    _below = str(featureName) + "_IsBelow_" + str(xName)
    _equ = str(featureName) + "_IsEqu_" + str(xName)
    
    abv = []
    below = []
    equ = []
    if (isinstance(x, np.ndarray) == False):
        for i in df[featureName]:
            if (i == x):
                equ.append(1)
            else:
                equ.append(0)

            if (i < x):
                below.append(1)
            else:
                below.append(0)
    
            if (i > x):
                abv.append(1)
            else:
                abv.append(0)
    else:
        for idx, item in enumerate(x):
            if (df[featureName][idx] == item):
                equ.append(1)
            else:
                equ.append(0)

            if (df[featureName][idx] < item):
                below.append(1)
            else:
                below.append(0)
    
            if (df[featureName][idx] > item):
                abv.append(1)
            else:
                abv.append(0)

    df[_above] = abv
    df[_below] = below
    df[_equ] = equ
    return df

def zeroFillAndReturnTrainAndTest(trainFeaturesDF, testFeaturesDF):
    featureUnion = list(set(testFeaturesDF.columns) | set(trainFeaturesDF.columns))
    featureUnion = sorted(featureUnion, key=str.lower)

    featureUnion.remove('dummy')
    featureUnion.insert(0, 'dummy')

    print(len(featureUnion), "Total Features:\n\t", featureUnion)
    print("")

    trainIsMissing = set(list(trainFeaturesDF.columns)).symmetric_difference(set(list(featureUnion)))
    print("Train is missing:\n\t", trainIsMissing)

    testIsMissing = set(list(testFeaturesDF.columns)).symmetric_difference(set(list(featureUnion)))
    print("Test is missing:\n\t", testIsMissing)
    print("")

    def fillMacro(df, missingColumns, featureUnion):
        for column in missingColumns:
            df[column] = np.zeros(df.shape[0])
        df = df[featureUnion] # Make sure all columns are the same order
        return df

    trainFeaturesDF = fillMacro(trainFeaturesDF, trainIsMissing, featureUnion)
    testFeaturesDF = fillMacro(testFeaturesDF, testIsMissing, featureUnion)

    print(trainFeaturesDF.head(2))
    print(testFeaturesDF.head(2))
    print("")
    return trainFeaturesDF.to_numpy(dtype=float), testFeaturesDF.to_numpy(dtype=float)

def oneHotAndJoin(df, columnName):
    oneHot = pd.get_dummies(df[columnName], prefix=str(columnName))
    df = df.join(oneHot)
    return df

def oneHotABunch(df):
    df = oneHotAndJoin(df, 'day')
    df = oneHotAndJoin(df, 'month')
    df = oneHotAndJoin(df, 'year')
    df = oneHotAndJoin(df, 'waterfront')
    df = oneHotAndJoin(df, 'view')
    df = oneHotAndJoin(df, 'condition')
    df = oneHotAndJoin(df, 'zipcode')
    df = oneHotAndJoin(df, 'yr_renovated')
    df = oneHotAndJoin(df, 'yr_built')
    df = oneHotAndJoin(df, 'floors')

    df = oneHotAndJoin(df, 'bathrooms')
    df = oneHotAndJoin(df, 'grade')
    return df

def normalizeNumerical(df):
    df['bedrooms'] = normalize(df['bedrooms'])           # normalize bedrooms
    df['floors'] = normalize(df['floors'])               # normalize floors

    df['waterfront'] = normalize(df['waterfront'])       # normalize waterfront
    df['view'] = normalize(df['view'])                   # normalize view
    df['condition'] = normalize(df['condition'])         # normalize condition

    df['bathrooms_norm'] = normalize(df['bathrooms'])         # normalize bathrooms
    df['bathrooms'] = np.log(df['bathrooms'].to_numpy(dtype=float))

    df['sqft_living_norm'] = normalize(df['sqft_living'])     # normalize sqft_living
    df['sqft_living'] = np.log(df['sqft_living'].to_numpy(dtype=float))

    df['grade_norm'] = normalize(df['grade'])                 # normalize grade
    df['grade'] = np.log(df['grade'].to_numpy(dtype=float))

    df['sqft_above_norm'] = normalize(df['sqft_above'])       # normalize sqft_above
    df['sqft_above'] = np.log(df['sqft_above'].to_numpy(dtype=float))

    df['sqft_living15_norm'] = normalize(df['sqft_living15']) # normalize sqft_living15
    df['sqft_living15'] = np.log(df['sqft_living15'].to_numpy(dtype=float))
    

    df['sqft_lot'] = normalize(df['sqft_lot'])           # normalize sqft_lot
    df['sqft_basement'] = normalize(df['sqft_basement']) # normalize sqft_basement
    df['yr_built'] = normalize(df['yr_built'])           # normalize yr_built
    df['yr_renovated'] = normalize(df['yr_renovated'])   # normalize yr_renovated
    df['zipcode'] = normalize(df['zipcode'])             # normalize zipcode
    df['lat'] = normalize(df['lat'])                     # normalize lat
    df['long'] = normalize(df['long'])                   # normalize long
    df['sqft_lot15'] = normalize(df['sqft_lot15'])       # normalize sqft_lot15

    df['month'] = normalize(df['month'])
    df['day'] = normalize(df['day'])
    df['year'] = normalize(df['year'])

    return df

def subtractFeatures(df, newFeatureName, featureAName, featureBName):
    df[newFeatureName] = df[featureAName] - df[featureBName]
    return df

def sumFeatures(df, newFeatureName, featureAName, featureBName):
    df[newFeatureName] = normalize(df[featureAName] + df[featureBName]) # New feature, new normal
    return df

def bigDuplicate(df, listOfPrefixFeatToDupe, dupeNumber):
    for featurePrefix in listOfPrefixFeatToDupe:
        currentNames = [x for x in df.columns.values.tolist() if x.startswith(featurePrefix)]
        for i in range(0, dupeNumber):
            for name in currentNames:
                _dupeName = 'copyOf' + featurePrefix + '-'+ name + '-'+ str(i)
                df[_dupeName] = df[name]
    return df

def getFeaturesAndGTPart3(csvFname, norm=True):
    df = pd.read_csv(csvFname)          # Fetches train data
    df = df.drop('id', 1)               # Part 0, a - "Remove ID Feature"
    df = splitDates(df)                 # Part 0, b - "Split the date feature into three separate numerical features: month, day , and year."
    
    # Normalize
    df = oneHotABunch(df)
    df = normalizeNumerical(df)
    
    df = bigDuplicate(df, ['sqft_living'], 40)
    df = bigDuplicate(df, ['sqft_above'], 40)
    df = bigDuplicate(df, ['bathrooms'], 40)

    gtNotLogNorm = df['price'].to_numpy(dtype=float)
    gt = np.log( df['price'].to_numpy(dtype=float) )   # price we're trying to predict (ground truth) and apply log transform
    df = df.drop('price', 1)
    
    features = df

    print("Fetched Features and GT of", csvFname, "\n\tNormalized: ", normalize, "\n\tFeatures:",features.shape, "\n\tGT:", gt.shape)
    return features, gt, gtNotLogNorm

def batchGradientDescent(learningRate, features, gt, reportName=False):
    w = np.zeros(features.shape[1], dtype=np.float64)
    n = gt.shape[0]
    _mse = []
    
    i = 0
    while True:
        gradient = (2 / n) * (np.dot( (np.dot(features, w) - gt), features))    # Calculate gradient
        w = np.subtract(w, np.dot(learningRate, gradient) )                     # Update weight
        theMSE = getMSE(features, gt, w)                                        # Calculate MSE
        norm = np.linalg.norm(gradient)                                         # Calculate norm of gradient
        _mse.append(theMSE)                                                     # record MSE
        
        print('\r[', i+1, "] [LR:", learningRate, "] Train MSE:", theMSE, "Norm:", norm,end=' ')
        
        i+=1
        if (norm < 0.03):
            break

    print("\r")

    if (reportName != False):
        saveReport(learningRate, reportName, _mse, "./part3.Output/")

    return w

def saveWeight(w):
    today = date.today()
    _name = str(today.year) + "." + str(today.month) + "." + str(today.day) + "." + "weight."+ str(random.randint(1111, 9999)) + ".txt"
    f = open("./part3.Output/" + _name, 'w')
    f.write(str(w) + "\n")
    f.close()
    print("Weight saved: ", _name)
    return

def getAllConditonIndex(npArr, val):
    indexArray = np.argwhere(npArr >= val)
    return indexArray

if __name__ == '__main__':
    print("\n\t>Part 3")

    learningRate = 0.0001
    
    trainFeaturesNormalizedDF, trainGT, gtNotNorm = getFeaturesAndGTPart3("./data/PA1_train.csv", True) # Prep training features normalized
    testFeaturesNormalizedDF, testGT, _ = getFeaturesAndGTPart3("./data/PA1_dev.csv", True)     # Prep testing features normalized
    
    trainFeaturesUnion, testFeaturesUnion = zeroFillAndReturnTrainAndTest(trainFeaturesNormalizedDF, testFeaturesNormalizedDF) # See report part 3 for why this does not leak training data to testing and vice-versa

    oversampleIdxArr = getAllConditonIndex(gtNotNorm, 10)
    for i in range(0, 10):
        overSampledNormTrainFeatArrUniond = np.squeeze(trainFeaturesUnion[oversampleIdxArr], axis=1)
        overSampledNormTrainGTArr = trainGT[oversampleIdxArr].flatten()
        trainFeaturesUnion = np.concatenate((trainFeaturesUnion, overSampledNormTrainFeatArrUniond))
        trainGT = np.concatenate((trainGT, overSampledNormTrainGTArr))

    w = batchGradientDescent(learningRate, trainFeaturesUnion, trainGT, reportName="PA1_train_Part3") 
    #w = np.zeros(trainFeaturesUnion.shape[1], dtype=np.float64)
    saveWeight(w)

    mse = getMSE(testFeaturesUnion, testGT, w)

    print("\nTest MSE: ", mse)
    print("\nRunning prediction on test/validation set...")

    dfTestPredictionResult = pd.DataFrame()
    dfTestPredictionResult['gt'] = pd.read_csv("./data/PA1_dev.csv")['price'].to_numpy(dtype=float)
    _prediction = []

    for idx, val in enumerate(testFeaturesUnion):
        _prediction.append(predict(w, val))
        #print("Prediction: ", predict(w, val), "\tGround Truth:", np.exp(testGT[idx]))
    
    dfTestPredictionResult['prediction'] = _prediction
    dfTestPredictionResult.to_csv("./part3.Output/testResult.csv")
    