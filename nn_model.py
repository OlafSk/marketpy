import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from keras.layers import Conv1D, Dense, Flatten, BatchNormalization, AveragePooling1D
from keras.optimizers import Adam



class nn_model():
    def __init__(self, steps_backward = 10, steps_forward = 2):
        self.model = Sequential()
        self.steps_backward = steps_backward
        self.steps_forward = steps_forward


    def yield_batches(self, X, y, steps_backward = None, steps_forward = None, scale=True):
       """
       :param X:
       :param y:
       :param steps_backward:
       :param steps_forward:
       :return:
       """
       if steps_forward == None:
           steps_forward = self.steps_forward
       if steps_backward == None:
           steps_backward = self.steps_backward
       X, y = X.copy(), y.copy()
       Xs = np.zeros((X.shape[0], steps_backward, X.shape[1]))
       ys = np.zeros((y.shape[0], ))
       i = 0
       while X.shape[0] >= steps_backward + steps_forward:
           index = np.random.randint(steps_backward, X.shape[0] - steps_forward, dtype=np.int32)
           if scale:
               Xs[i] = StandardScaler().fit_transform(X[index - steps_backward:index])
               ys[i] = (y[index + steps_forward] / y[index]) - 1
           else:
               Xs[i] = X[index - steps_backward:index]
               ys[i] = (y[index + steps_forward] / y[index]) - 1
           X = np.delete(X, np.linspace(index - steps_backward, index, dtype=np.int32), 0)
           y = np.delete(y, np.linspace(index, index + steps_forward, dtype=np.int32), 0)
           i += 1
       return Xs[:i], ys[:i]

    def yield_batches_from_all(self, df):

        samples, targets = [], []
        for indice in np.unique(df['indice']):
            data = df[df.indice == indice]
            x, y = data.loc[:, ['close']].values, data.loc[:, ['close']].values
            sample, target = self.yield_batches(x, y, 5000)
            samples.append(sample); targets.append(target)
        return np.concatenate(samples, axis=0), np.concatenate(targets, axis=0)


    def yield_rolling_prediction(self, X, y):
        """
        :param X:
        :return:
        """
        X = X.copy()
        samples = np.zeros((X.shape[0] - self.steps_backward - self.steps_forward, self.steps_backward, 1))
        targets = np.zeros((X.shape[0] - self.steps_backward - self.steps_forward,))
        for _ in range(X.shape[0] - self.steps_backward - self.steps_forward):
            samples[_] = StandardScaler().fit_transform(X[_:_+self.steps_backward])
            targets[_] = (y[_+self.steps_backward+self.steps_forward] / X[_+self.steps_backward]) - 1
        return samples, targets



df = pd.read_csv('data/indicies/all_indices.csv')




nn = nn_model(steps_forward = 3)

X, y = nn.yield_batches_from_all(df)



nn.model.add(Conv1D(1, 3, activation='relu', input_shape=(10, 1)))
nn.model.add(BatchNormalization())
nn.model.add(Conv1D(1, 3, activation='relu'))
nn.model.add(BatchNormalization())
nn.model.add(AveragePooling1D())
nn.model.add(Flatten())
nn.model.add(Dense(1))


nn.model.compile("adam", "mean_squared_error")


nn.model.fit(X,y, batch_size = 32)
