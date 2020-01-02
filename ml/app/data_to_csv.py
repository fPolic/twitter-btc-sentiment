import pandas as pd
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

# Normalize


def normalize(dev, df, column):

    dev[column] = (df[column] - df[column].min()) / \
        (df[column].max() - df[column].min())

# Standardize


def calculateZScore(dev, df, column):

    dev[column] = (df[column] - df[column].rolling(WINDOW_SIZE).mean()
                   )/df[column].rolling(WINDOW_SIZE).std(ddof=0)


def calculateShareOfEmotion(share, df, column):
    share[column] = (df[column]/df['count'])


def main():

    btc = getBTCDataFrame()
    btc.set_index('date')

    emotions = getEmotionsDataFrame()
    emotions.set_index('date')

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

    btc.to_csv('timeseries/btc.csv', index=False)
    dev.to_csv('timeseries/dev.csv', index=False)
    share.to_csv('timeseries/share.csv', index=False)


if __name__ == "__main__":
    main()
