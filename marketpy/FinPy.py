import pandas as pd
import numpy as np
from keras.layers import Dense, LSTM, BatchNormalization, Dropout, Conv1D
import keras
from keras.models import Sequential
from keras.losses import mean_squared_error
from sklearn.metrics import auc
from keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick2_ohlc
from keras.models import load_model
#import talib

def scale(X):
    mean = np.mean(X)
    std = np.std(X)
    return (X - mean) / std

def plotOHLCfromData(X, width = 0.6):
    fig, ax = plt.subplots()
    candlestick2_ohlc(ax,X[:,0],X[:,1], X[:,2],X[:,3],width=0.6)
    plt.show()

def meanReturnCurve(y_true, y_pred):
    thresholds = np.unique(y_pred)
    meanReturn = np.empty((len(thresholds)))
    percent = np.empty((len(thresholds)))
    for i, thresh in enumerate(thresholds):
        meanReturn[i] = np.mean(y_true[y_pred >= thresh])
        percent[i] = sum(y_pred >= thresh)/len(y_pred)
    return auc(percent, meanReturn)

def plotMeanReturnCurve(y_true, y_pred, **kwargs):
    thresholds = np.unique(y_pred)
    meanReturn = np.empty((len(thresholds)))
    percent = np.empty((len(thresholds)))
    for i, thresh in enumerate(thresholds):
        meanReturn[i] = np.mean(y_true[y_pred >= thresh])
        percent[i] = sum(y_pred >= thresh)/len(y_pred)
    return plt.plot(percent, meanReturn, **kwargs)

def quantileScore(y_true, y_pred, percent = 80):
    max_return = y_true[y_true > np.percentile(y_true, percent)].mean()
    model_return = y_true[y_pred > np.percentile(y_pred, percent)].mean()
    return model_return/max_return

def plotPercentageOver0(y_true, y_pred, **kwargs):
    thresholds = np.unique(y_pred)
    meanReturn = []
    percent = []
    for thresh in thresholds:
        meanReturn.append(np.mean(y_true[y_pred >= thresh] > 0))
        percent.append(sum(y_pred >= thresh)/len(y_pred))
    return plt.plot(percent, meanReturn, **kwargs)

def train_test_split(X, y, test_size = 0.3):
    assert len(X) == len(y), "X and y not the same size"
    size = int((1 - test_size) * len(X))
    X_train = X[:size]
    X_test = X[size:]
    y_train = y[:size].values.reshape(-1,1)
    y_test = y[size:].values.reshape(-1,1)
    return X_train, X_test, y_train, y_test

def prepareDataFromStooq(df):
    if 'Wolumen' in df.columns:
        df.drop(['Data', 'Wolumen', 'LOP'], inplace = True, axis=1)
    else:
        df.drop('Data', inplace=True, axis = 1)
    y = df['Zamkniecie'].shift(-5) - df['Otwarcie']
    df = df.iloc[:-5,:]
    y = y[:-5]
    return df.values, y

def downloadDataFromStooq(link):
    df = pd.read_csv(link)
    return df
def AddTechnicalFeatures(X):
    #0 - open
    #1 - High
    #2 - Low
    #3 - Close
    k, dfast = talib.STOCH(X[1],X[2],X[3])
    print(k.shape)
    print(X.shape)
    X = np.hstack((X, k))
    return X
def CheckForInwestment(model, link):
    df = downloadDataFromStooq(link)
    data = df['Data'].values[-1]
    X,y = prepareDataFromStooq(df)
    C = PrepareDataForPrediction(X)
    X_train, X_test, y_train, y_test = test_train_split_for_lstm(X, y,
                                                                test_size=0.4)
    spy_model = load_model(model)
    y_pred = spy_model.predict(X_test)
    percentyl = round(np.percentile(y_pred, 80),4)
    pred = round(float(spy_model.predict(C)),4)
    return pred, percentyl, data

def PrepareDataForPrediction(X, time_step = 10):
    V = np.empty((1, time_step, X.shape[1]))
    V[0,:,:] = scale(X[-time_step:,:])
    return V
##tutaj jest zle powinno brac 10 w do tylu a nie do prozdu
def prepareDataForLSTM(X_train, X_test,y_train, y_test, time_step = 10):

    V_test = np.empty([len(X_test) - time_step,time_step,X_train.shape[1]])
    V_train = np.empty([len(X_train) - time_step,time_step,X_train.shape[1]])
    for i in range(time_step, len(X_test)):
        V_test[i - time_step,:,:4] = scale(X_test[i - time_step:i, :4])
        V_test[i - time_step,:, 4:] = X_test[i - time_step:i, 4:]
    for i in range(time_step, len(X_train)):
        V_train[i - time_step,:,:4] = scale(X_train[i - time_step:i,:4])
        V_train[i - time_step,:, 4:] = X_train[i - time_step:i, 4:]
    y_train = y_train[time_step:]
    y_test = y_test[time_step:]
    return V_train, V_test, y_train, y_test

def test_train_split_for_lstm(X,y, test_size=0.3, time_step = 10):
    X_train, X_test, y_train, y_test = test_train_split(X, y, test_size=test_size)
    X_train, X_test, y_train, y_test = prepareDataForLSTM(X_train, X_test,
                                        y_train, y_test, time_step = time_step)
    return X_train, X_test, y_train, y_test
