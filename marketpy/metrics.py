import numpy as np
from sklearn.metrics import auc

def quantile_score(y_true, y_pred, percent = 80):
    """
    Calculates the "quantile score" defined as mean of true returns where
    prediction is the highest 20 percentile.

    Keyword arguments:
    y_true -- numpy array of true returns
    y_pred -- numpy array of predicted returns
    percent -- percent of highest anserws taken into account
    """
    max_return = y_true[y_true > np.percentile(y_true, percent)].mean()
    model_return = y_true[y_pred > np.percentile(y_pred, percent)].mean()
    return model_return/max_return


def mean_return_curve(y_true, y_pred):
    """
    Calulates the area under mean return curve.
    Mean return curve is calulated by taking highest numer of predictions
    and calculating mean return from true returns curve based on choosen
    predcitions.
    Keyword arguments:
    y_true -- numpy array of true returns
    y_pred -- numpy array of predicted returns
    """
    thresholds = np.unique(y_pred)
    mean_return = np.empty((len(thresholds)))
    percent = np.empty((len(thresholds)))
    for i, thresh in enumerate(thresholds):
        mean_return[i] = np.mean(y_true[y_pred >= thresh])
        percent[i] = sum(y_pred >= thresh)/len(y_pred)
    return auc(percent, mean_return)

def percentage_over_zero(y_true, y_pred, percent = 80):
    """
    Calculates the percentage of true returns over zero, when taken where
    y_pred is bigger then y_pred percentile.
    Keyword arguments:
    y_true -- numpy array of true returns
    y_pred -- numpy array of predicted returns
    percent -- percent of highest anserws taken into account
    """
    y_choosen = y_true[y_pred > np.percentile(y_pred,percent)]
    return sum(y_choosen > 0) / len(y_choosen)
