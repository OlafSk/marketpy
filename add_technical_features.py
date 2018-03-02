import pandas as  pd
import marketpy
import numpy as np
import os
from importlib import reload
reload(marketpy)
import talib
import inspect


def add_candle_patterns(X, y, return_array = False):
    """
    Adds candle patterns used in technical analysis.
    Keyword arguments:
    X - dataframe contaning predictors where cols:
    #0 - open
    #1 - High
    #2 - Low
    #3 - Close
    y - vector of returns.
    """
    data = {
            "open" : X.iloc[:,0].values,
            "high" : X.iloc[:,1].values,
            "low" : X.iloc[:,2].values,
            "close" : X.iloc[:,3].values
    }
    for func in talib.__dict__.keys():
        if func[:3] == "CDL":
            X[func] = talib.__dict__[func](**data)

    return X

def add_all_technical_indicators(X, y):
    """
    Adds candle patterns used in technical analysis.
    Keyword arguments:
    X - dataframe contaning predictors where cols:
    #0 - open
    #1 - High
    #2 - Low
    #3 - Close
    y - vector of returns.
    """
    data = {
            "open" : X.iloc[:,0].values,
            "high" : X.iloc[:,1].values,
            "low" : X.iloc[:,2].values,
            "close" : X.iloc[:,3].values
    }
    for func in talib.__dict__.keys():
        if func.isupper():
            if func[:3] != "CDL":
                try:
                    X[func] = talib.__dict__[func](**data)
                except Exception as e:
                    print(e)
    return X



for file in os.listdir("data"):
    print(file)
    df = pd.read_csv("data/" + file)

    X, y = marketpy.prepare_data_from_stooq(df)
    X, y = marketpy.preprocessing.add_technical_features(X, y, return_array=False)
    X = pd.DataFrame(X)
    X["y"] = y
    X.to_csv("ta/" + file[:-3] + "_ta.csv")
