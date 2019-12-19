import fire
import pandas as pd
import tensorflow as tf

from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.models import Model

import matplotlib.pyplot as plt

from numpy import array
from pymongo import MongoClient

DATABASE_NAME = 'fer'
mongoURL = 'mongodb://localhost:27017/'
# global Mongo connection
DB = MongoClient(mongoURL)[DATABASE_NAME]

WINDOW_SIZE = 24
EMOTIONS = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative',
            'positive', 'sadness', 'surprise', 'trust', 'count']

# =================== DATA PREPARATION ===================


def getEmotionsDataFrame():
    collection = DB["emotions_aggregation"]
    cursor = list(collection.find(no_cursor_timeout=True))

    return pd.DataFrame(cursor)


def getBTCDataFrame():
    collection = DB["btc"]
    cursor = list(collection.find(no_cursor_timeout=True))

    df = pd.DataFrame(cursor, columns=['date', 'close', 'volume', 'open'])
    df['return'] = df['close'] / df['open'] - 1

    return df


def calculateZScore(dev, df, column):
    # TODO: CHECK IF JOIN IS CORRECT
    dev[column] = (df[column] - df[column].rolling(WINDOW_SIZE).mean()
                   )/df[column].rolling(WINDOW_SIZE).std(ddof=0)


def calculateShareOfEmotion(share, df, column):
    share[column] = (df[column]/df['count']).rolling(WINDOW_SIZE).mean()

# =================== PLOTING TIME SERIES ===================


def render(share, dev, btc, plot_args):
    btc = btc.merge(dev[['count', 'date']], on='date', how='inner')

    if (plot_args == True):
        share.plot(x="date", figsize=(14, 8),
                   title='Emotion word count share in total count', fontsize=12)
        dev.head(200).plot(x="date", subplots=True, layout=(6, 2), figsize=(14, 8),
                           title='Rolling z-score for 24h period', fontsize=12)
        btc.plot(x="date", y=["close", 'volume', 'return', 'count'],
                 subplots=True, layout=(4, 1), figsize=(8, 8),
                 title='BTC data', fontsize=12)

    if (type(plot_args) != bool):
        if ('btc' in plot_args):
            btc.plot(x="date", y=["close", 'volume', 'return', 'count'],
                     subplots=True, layout=(4, 1), figsize=(8, 8),
                     title='BTC data', fontsize=12)
        if ('dev' in plot_args):
            dev.head(200).plot(x="date", subplots=True, layout=(6, 2), figsize=(14, 8),
                               title='Rolling z-score for 24h period', fontsize=12)
        if ('share' in plot_args):
            share.plot(x="date", figsize=(14, 8),
                       title='Emotion word count share in total count', fontsize=12)

    plt.show()

# =================== LSTM MODEL ===================


def trainModel(test_data):
    X = [X[0] for X in test_data]
    Y = [X[1] for X in test_data]

    T = array(X[-1]).reshape(1, 1, 1)
    X = array(X[:-1]).reshape(199, 1, 1)

    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(1, 1)),
        LSTM(50, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])
    model.fit(X, Y[:-1], epochs=500, validation_split=0.2, batch_size=5)

    baseline = pd.DataFrame(Y).tail(24).mean()
    print(model.predict(T, verbose=True), Y[-1], baseline)
    return model


# =================== MAIN CLI APP ===================

def main(plot=None, train=None, window=WINDOW_SIZE):
    global WINDOW_SIZE
    WINDOW_SIZE = window  # optional param overrides default

    btc = getBTCDataFrame()
    emotions = getEmotionsDataFrame()
    dev = emotions[['date']]
    share = emotions[['date']]

    # =================== PREPROCESS DATA ===================

    for em in EMOTIONS:
        calculateZScore(dev, emotions, em)
        if em == 'count':
            continue
        calculateShareOfEmotion(share, emotions, em)

    dev.dropna(axis="rows", inplace=True)
    share.dropna(axis="rows", inplace=True)

    # btc.to_csv('./colab/btc.csv', index=False)
    # emotions.to_csv('./colab/emotions.csv', index=False)
    # dev.to_csv('./colab/dev.csv', index=False)
    # share.to_csv('./colab/share.csv', index=False)

    if plot is not None:
        render(share, dev, btc, plot)

    if train is not None:

        test_data = dev.merge(btc, on="date", how="inner")[
            ['fear', 'return']].head(200).values

        model = trainModel(test_data)
        print(model.summary())


if __name__ == "__main__":
    """
      CLI APP example commands:
        1. python3 main.py --plot --window=24
        2. python3 main.py --plot=btc,share
        3. python3 main.py --train
    """
    fire.Fire(main)
