import numpy as np
import pandas as pd

# Parameters: data df
# Output: data but with date column dropped, dates are split into their own month / day / year columns
def splitDates(df):
    allDates = df['date']
    month = []
    day = []
    year = []
    
    for i in allDates:
        _data = i.split("/")
        month.append(int(_data[0]))
        day.append(int(_data[1]))
        year.append(int(_data[2]))

    df['month'] = month
    df['day'] = day
    df['year'] = year

    df = df.drop('date', 1)
    return df

def report(df, fname):
    fname = fname.split("/")[-1].replace(".csv","")
    _name = []
    _type = []
    _mean = []
    _stdev = []
    _range = []
    _percentage = []

    numericalFeatures = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
    categoricalFeatures = ['waterfront', 'view', 'grade', 'condition']
    
    for item in numericalFeatures:
        _name.append(item)
        _type.append('numerical')
        _mean.append(np.mean(df[item]))
        _stdev.append(np.std(df[item]))
        _range.append([np.amin(df[item]),np.amax(df[item])])
        _percentage.append(None)

    totalDFLen = len(df.index)

    for item in categoricalFeatures:
        _name.append(item)
        _type.append('categorical')
        _mean.append(None)
        _stdev.append(None)
        _range.append(None)
        
        unique, counts = np.unique(df[item], return_counts=True)
        _arr = np.asarray((unique, counts)).T
        _finalArr = []
        for idx, j in enumerate(_arr):
            _finalArr.append((j[0], str(round(float(j[1] / totalDFLen), 3) * 100) + "%", str(j[1]) + "/" + str(totalDFLen)))

        _percentage.append(_finalArr)
    
    report = pd.DataFrame()
    report['name'] = _name
    report['type'] = _type
    report['mean'] = _mean
    report['stdev'] = _stdev
    report['range'] = _range
    report['percentage'] = _percentage
    
    #report.to_csv(fname + ".report.csv", index=False) # Saves report to local directory
    return report

# Numerically normalizes a given column
def normalize(column):
    _max = np.amax(column)
    _min = np.amin(column)
    _denom = _max - _min
    _new = []

    for i, val in enumerate(column):
        _new.append( (val - _min) / (_denom) )

    return _new

# Parameters: data df
# Output: normalized columns
def normalizeNumerical(df):
    df['bedrooms'] = normalize(df['bedrooms'])           # normalize bedrooms
    df['bathrooms'] = normalize(df['bathrooms'])         # normalize bathrooms
    df['sqft_living'] = normalize(df['sqft_living'])     # normalize sqft_living
    df['sqft_lot'] = normalize(df['sqft_lot'])           # normalize sqft_lot
    df['floors'] = normalize(df['floors'])               # normalize floors

    df['waterfront'] = normalize(df['waterfront'])       # normalize waterfront
    df['view'] = normalize(df['view'])                   # normalize view
    df['condition'] = normalize(df['condition'])         # normalize condition
    df['grade'] = normalize(df['grade'])                 # normalize grade

    df['sqft_above'] = normalize(df['sqft_above'])       # normalize sqft_above
    df['sqft_basement'] = normalize(df['sqft_basement']) # normalize sqft_basement
    df['yr_built'] = normalize(df['yr_built'])           # normalize yr_built
    df['yr_renovated'] = normalize(df['yr_renovated'])   # normalize yr_renovated
    df['zipcode'] = normalize(df['zipcode'])             # normalize zipcode
    df['lat'] = normalize(df['lat'])                     # normalize lat
    df['long'] = normalize(df['long'])                   # normalize long
    df['sqft_living15'] = normalize(df['sqft_living15']) # normalize sqft_living15
    df['sqft_lot15'] = normalize(df['sqft_lot15'])       # normalize sqft_lot15

    df['month'] = normalize(df['month'])
    df['day'] = normalize(df['day'])
    df['year'] = normalize(df['year'])

    return df

def part1(fname):
    df = pd.read_csv(fname)     # Fetches data

    df = df.drop('id', 1)       # Part 0, a - "Remove ID Feature"
    print("\n\t> Part 0, a - Remove ID feature")
    print(df)

    df = splitDates(df)         # Part 0, b - "Split the date feature into three separate numerical features: month, day , and year."
    print ("\n\t> Part 0, b - Split Month/Day/Year into their own columns and drop date column.")
    print(df)

    _report = report(df, fname) # Part 0, c - "Build a table that reports the statistics for each feature. For numerical features, please report the mean, 
                                #              the standard deviation, and the range. For categorical features such as waterfront, grade, condition
                                #              (the later two are ordinal), please report the percentage of examples for each category."
    print("\n\t> Part 0, c - Statistics for each Feature")
    print(_report)

    df = normalizeNumerical(df) # Part 0, e - "Normalize all numerical features (excluding the housing prices y) to the range 0 and 1 using the training data."
    print("\n\t> Part 0, e - Normalized Data")
    print(df)
    df.to_csv("PA1_train.norm.csv", index=False)
    return df
    
if __name__ == '__main__':
    part1Data = part1("./data/PA1_train.csv")   # Fetches training data from /data/PA1_Train.csv
