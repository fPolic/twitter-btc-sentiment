import fire
import datetime
import pandas as pd

from numpy import array
from pymongo import MongoClient
from plotly.subplots import make_subplots
from plotly import offline

from models.lstm import train as trainLSTM
from models.xgboost__ import train as trainXGBoost


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
    share[column] = (df[column]/df['count'])  # .rolling(WINDOW_SIZE).mean()

# =================== PLOTING TIME SERIES ===================


def render(share, dev, btc):
    btc = btc.merge(dev[['count', 'date']], on='date', how='right')

    fig = make_subplots(rows=4, cols=1,  shared_xaxes=True, subplot_titles=(
        'Rolling z-score for 24h window', 'Emotion word count share in total count', 'Bitcoin price', 'Bitcoin volume'))

    # ==================== DEVIATIONS ====================
    for em in EMOTIONS:
        fig.add_scatter(x=dev['date'], y=dev[em],
                        mode='lines', row=1, col=1, name="Standard dev./" + em)

    # ==================== SHARES ====================
    for em in EMOTIONS[:-1]:
        fig.add_scatter(x=share['date'], y=share[em],
                        mode='lines', row=2, col=1, name="Share/" + em)

    # ==================== BITCOIN ====================
    fig.add_scatter(x=btc['date'], y=btc['close'],
                    mode='lines', row=3, col=1, name='Bitcoin close price (hourly)')
    fig.add_bar(x=btc['date'], y=btc['volume'],
                row=4, col=1, name='Bitcoin volume (hourly)')
    fig.update_layout(title_text="BTC tweets lexicon analysis")

    offline.plot(fig, filename='timeseries.html', auto_open=True)

# =================== MAIN CLI APP ===================


def main(plot=None, train=None, window=WINDOW_SIZE):
    global WINDOW_SIZE
    WINDOW_SIZE = window  # optional param overrides default

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

    if plot is not None:
        render(share, dev, btc)

    if train is not None:

        test_data = share.merge(btc, on="date", how="inner")[
            EMOTIONS[:-1] + ['date'] + ['return', 'close']]
        model = (trainXGBoost if train == 'xgboost' else trainLSTM)(test_data)


if __name__ == "__main__":
    """
      CLI APP example commands:
        1. python3 main.py --plot --window=24
        2. python3 main.py --plot=btc,share
        3. python3 main.py --train
    """
    fire.Fire(main)
