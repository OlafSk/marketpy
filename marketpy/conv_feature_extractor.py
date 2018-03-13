import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/all_indices.csv')

X = df.loc[:, ['close']].values
y = df.loc[:, ['close']].values

def yield_batches(X, y, steps_backward, steps_forward, batch_size):
    """

    :param X:
    :param y:
    :return:
    """
    X, y = X.copy(), y.copy()
    samples = np.zeros((batch_size, steps_backward, 1))
    targets = np.zeros((batch_size,))
    for _ in range(batch_size):
        index = np.random.randint(steps_backward, X.shape[0] - steps_forward - 1, size=1)[0]
        samples[_] = StandardScaler().fit_transform(X[index - steps_backward:index])
        targets[_] = (y[index + steps_forward] / X[index]) - 1
        X = np.delete(X, np.linspace(index - steps_backward, index, dtype=np.int32), 0)
        y = np.delete(y, np.linspace(index, index + steps_forward, dtype=np.int32), 0)
        if X.shape[0] <= steps_backward + steps_forward + 1:
            break
    return samples[:_], targets[:_]


def yield_rolling_prediction(X, y, steps_backward=30, steps_forward=5):
    """

    :param X:
    :return:
    """
    X = X.copy()
    samples = np.zeros((X.shape[0] - steps_backward - steps_forward, steps_backward, 1))
    targets = np.zeros((X.shape[0] - steps_backward - steps_forward,))
    for _ in range(X.shape[0] - steps_backward - steps_forward):
        samples[_] = StandardScaler().fit_transform(X[_:_+steps_backward])
        targets[_] = (y[_+steps_backward+steps_forward] / X[_+steps_backward]) - 1
    return samples, targets

Xi, yi = yield_batches(X, y, 30, 5, 1000)

from keras.models import Sequential
from keras.layers import Conv1D, Dense, Flatten
from keras.optimizers import Adam

adam = Adam(lr=0.001)

model = Sequential()
model.add(Conv1D(12, 3, activation='relu', input_shape=(30, 1)))
model.add(Flatten())
model.add(Dense(1))

model.compile(adam, 'mean_squared_error')
model.fit(Xi, yi, batch_size=32, epochs=15)

Xrp, yrp = yield_rolling_prediction(X, y, 30)

preds = model.predict(Xrp)

import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize=(12, 8))
ax.plot(yrp, c='blue')
ax.plot(preds, c='red')