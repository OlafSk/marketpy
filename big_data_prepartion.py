from marketpy.preprocessing import *
import pandas as pd
import os


df_all = pd.DataFrame()
for file_name in os.listdir("data/indicies"):
    df = pd.read_csv("data/indicies/" + file_name)
    X, y =prepare_data_from_stooq(df)
    X, _, y, _ = train_test_split(X,y, test_size = 0.2)

    columns = ['open','high','low','close']

    df = pd.DataFrame(X, columns = columns)

    df['5_day_return'] = y

    df['indice'] = file_name[:-4]
    df_all = pd.concat([df_all,df])
    df_all.reset_index(inplace=True, drop=True)


def prepare_data_for_lstm(X,y, time_step = 20):
    """


    """
    V = np.empty([X.shape[0] - time_step, time_step, X.shape[1]])
    for i in range(time_step, X.shape[0]):
        V[i - time_step, :, X.shape[1]]
    y = y[time_step:]
    return V, y

prepare_data_for_lstm(X, y)
