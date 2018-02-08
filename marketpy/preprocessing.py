import pandas as pd
import numpy as np
import talib


def train_test_split(X, y, test_size = 0.3):
    """
    Returns data split in train and test part.
    Test part cotains the last 0.3 percentage of data, while train part
    contains rest. Useful in spliting time series data, where we want to predict
    on data that model has never seen
    Keyword arguments:
    X -- data frame or numpy array contaning predictors
    y -- dataframe or numpy array contaning predicted values
    test_size -- percent of data taken into test sample
    """
    assert len(X) == len(y), "X and y not the same size"
    size = int((1 - test_size) * len(X))
    X_train = X[:size]
    X_test = X[size:]
    y_train = y[:size].values.reshape(-1,1)
    y_test = y[size:].values.reshape(-1,1)
    return X_train, X_test, y_train, y_test


def prepare_data_from_stooq(df, return_days = 5):
    """
    Prepares data for X, y format from pandas dataframe
    downloaded from stooq. Y is created as closing price in return_days
    - opening price
    Keyword arguments:
    df -- data frame contaning data from stooq
    return_days -- number of day frame in which to calculate y.
    """
    if 'Wolumen' in df.columns:
        df.drop(['Data', 'Wolumen', 'LOP'], inplace = True, axis=1)
    else:
        df.drop('Data', inplace=True, axis = 1)
    y = df['Zamkniecie'].shift(-return_days) - df['Otwarcie']
    df = df.iloc[:-return_days,:]
    y = y[:-return_days]
    return df.values, y


def add_technical_features(X, y):
    """
    Adds basic technical features used in paper:
    "https://arxiv.org/pdf/1706.00948.pdf" using library talib.
    Keyword arguments:
    X -- numpy array or dataframe contaning predictors where cols:
        #0 - open
        #1 - High
        #2 - Low
        #3 - Close
    y -- vector of returns.
    """
    k, dfast = talib.STOCH(X[:,1],X[:,2],X[:,3])
    X = np.hstack((X, k.reshape(-1,1)))
    X = np.hstack((X, dfast.reshape(-1,1)))
    X = np.hstack((X, talib.SMA(dfast, timeperiod=5).reshape(-1,1)))
    X = np.hstack((X, talib.MOM(X[:,3], timeperiod=4).reshape(-1,1)))
    X = np.hstack((X, talib.ROC(X[:,3], timeperiod=5).reshape(-1,1)))
    X = np.hstack((X, talib.WILLR(X[:,1], X[:,2], X[:,3],
                                        timeperiod=5).reshape(-1,1)))
    X = np.hstack((X, (X[:,3] / talib.SMA(X[:,3], timeperiod=5)).reshape(-1,1)))
    X = np.hstack((X, (X[:,3] / talib.SMA(X[:,3], timeperiod=10)).reshape(-1,1)))
    X = np.hstack((X, talib.RSI(X[:,3]).reshape(-1,1)))
    X = np.hstack((X, talib.CCI(X[:,1], X[:,2], X[:,3],
                                        timeperiod=14).reshape(-1,1)))
    X = X[~np.isnan(X).any(axis = 1)]
    y = y[-X.shape[0]:]
    return X, y
