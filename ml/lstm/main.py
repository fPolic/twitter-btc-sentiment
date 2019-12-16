import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pymongo import MongoClient

DATABASE_NAME = 'fer'
mongoURL = 'mongodb://localhost:27017/'
# global Mongo connection
DB = MongoClient(mongoURL)[DATABASE_NAME]

WINDOW_SIZE = 24
EMOTIONS = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative',
            'positive', 'sadness', 'surprise', 'trust', 'count']


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


def calculateZScore(dev, df, column, window=WINDOW_SIZE):
    # TODO: CHECK IF JOIN IS CORRECT
    dev[column] = (df[column] - df[column].rolling(window).mean()
                   )/df[column].rolling(window).std(ddof=0)


def calculateShareOfEmotion(share, df, column):
    share[column] = (df[column]/df['count']).rolling(WINDOW_SIZE).mean()


def plot(share, dev, btc):

    share.plot(x="date")
    dev.head(200).plot(x="date", subplots=True, layout=(6, 2))
    btc.plot(x="date", y=["close", 'volume', 'return', 'count'],
             subplots=True, layout=(4, 1))

    plt.show()


def main():
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

    dev = dev.dropna(axis="rows")
    # btc = btc.merge(dev[['count', 'date']], on='date', how='inner')
    # plot(share, dev, btc)

    test_data = dev.merge(btc, on="date", how="inner")[
        ['fear', 'return']].head(200).values

    # =================== DEFINE MODEL ===================

    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(1, 1)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())

    # =================== TRAIN MODEL ===================


if __name__ == "__main__":
    main()
