import pandas as pd
import numpy as np
import talib


def load_data(ticker):
    """

    """
    path_to_data = 'https://stooq.pl/q/d/l/?s={ticker}&i=d'.format(
        ticker=ticker)
    return pd.read_csv(path_to_data)


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


def prepare_data_from_stooq(df, to_prediction = False, return_days = 5):
    """
    Prepares data for X, y format from pandas dataframe
    downloaded from stooq. Y is created as closing price in return_days
    - opening price
    Keyword arguments:
    df -- data frame contaning data from stooq
    return_days -- number of day frame in which to calculate y.
    """
    if 'Wolumen' in df.columns:
        df = df.drop(['Data', 'Wolumen', 'LOP'], axis=1)
    else:
        df = df.drop('Data', axis = 1)
    y = df['Zamkniecie'].shift(-return_days) - df['Otwarcie']
    if not to_prediction:
        df = df.iloc[:-return_days,:]
        y = y[:-return_days]/df['Otwarcie']
    return df.values, y


def add_technical_features(X, y, return_array = False):
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
    y = y[~np.isnan(X).any(axis = 1)]
    X = X[~np.isnan(X).any(axis = 1)]
    if return_array:
        return X, y
    else:
        colnames = ['open','high','low','close','stoch_k', 'stoch_d', 'SMA_5', 'mom', 'roc', 'willr', 'disp_5','disp_10','rsi','cci']
        return pd.DataFrame(X, columns=colnames), y

    return X, y


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
                X[func] = talib.__dict__[func](**data)
    return X

def prepare_data_for_lstm(X,y, time_step = 20, normalize = True):
    """
    Prepares data for learning RNN
    """
    V = np.empty([X.shape[0] - time_step, time_step, X.shape[1]])
    for i in range(time_step, X.shape[0]):
        time_data = X[i - time_step:i, :]
        if normalize:
            time_data = (time_data - np.mean(time_data, axis = 1, keepdims = True))/ np.std(time_data,axis = 1, keepdims = True)
        V[i - time_step, :, :] = time_data

    y = y[time_step:]
    return V, y
