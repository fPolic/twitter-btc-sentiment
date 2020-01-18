import fire
import datetime
import pandas as pd

from numpy import array
from pymongo import MongoClient

from pandas import Grouper

from plotly import offline
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

from models.lstm import train as trainLSTM
from models.xgboost__ import train as trainXGBoost

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

    _emotion = pd.DataFrame(cursor)
    _emotion.set_index("date", inplace=True)
    _emotion.index = pd.to_datetime(_emotion.index)

    return _emotion[EMOTIONS]


def calculateStandardDev(dev, df, column):
    dev[column] = (df[column]-df[column].mean())/df[column].std(ddof=0)

# =================== PLOTING TIME SERIES ===================


def render(emotions, share, dev, btc):

    # ======================================== BOXPLOTS ========================================

    fig = make_subplots(rows=5, cols=2, subplot_titles=[
                        x for x in EMOTIONS[:-1]])

    row = 0
    for em in EMOTIONS[:-1]:
        row = row + 1
        for key, group in emotions.groupby([emotions.index.year, emotions.index.month]):
            # anger[key] = group['anger'].values
            fig.add_trace(go.Box(
                y=group[em].values,
                name=str(key),
            ), row=row % 5 + 1, col=row % 2 + 1)

    fig.update_layout(
        title_text="Emotions boxplots monthly", showlegend=False, height=1500)

    offline.plot(fig, filename='static/boxplots.html', auto_open=True)

    # ======================================== OVERVIEW ========================================

    fig = make_subplots(rows=2,  shared_xaxes=True, subplot_titles=(
        'Emotional word count per hour', 'Bitcoin price'))

    for em in EMOTIONS[:-1]:
        fig.add_scatter(x=emotions.index, y=emotions[em],
                        mode='lines', row=1, col=1, name=em)

    btc = btc.merge(emotions[[]], right_index=True,
                    left_index=True, how='right')

    fig.add_scatter(x=btc.index, y=btc['close'],  col=1,
                    mode='lines', row=2, name='Bitcoin close price (hourly)')

    fig.update_layout(title_text="Overview")
    offline.plot(fig, filename='static/overview.html', auto_open=True)

    # ======================================== EMOTIONS ========================================

    fig = make_subplots(rows=3,  shared_xaxes=False, subplot_titles=(
        'Emotion word count share in total count', 'Emotions standard deviations', "Emotions box plot"))

    for em in EMOTIONS[:-1]:

        fig.add_scatter(x=share.index, y=share[em],
                        mode='lines', row=1, col=1, name="Share/" + em, legendgroup="Share")

        fig.add_scatter(x=dev.index, y=dev[em],
                        mode='lines', row=2, col=1, name="Standard dev./" + em, legendgroup="Dev")

        fig.add_box(y=share[em],
                    row=3, col=1, name="Box/" + em, legendgroup="Boxplot")

    fig.update_layout(title_text="Emotion share")
    offline.plot(fig, filename='static/emotions.html', auto_open=True)

    # ======================================== DISTRIBUTION ========================================

    fig = make_subplots(rows=3,  shared_xaxes=False,
                        subplot_titles=('Emotions histogram'))

    fig = ff.create_distplot(
        [share[c] for c in EMOTIONS[:-1]], EMOTIONS[:-1], bin_size=0.0)

    fig.update_layout(
        title_text="Emotions distribution histogram", legend_orientation="h")

    offline.plot(fig, filename='static/histogram.html', auto_open=True)


# =================== MAIN CLI APP ===================


def main(plot=None, train=None, serve=False, window=WINDOW_SIZE):
    global WINDOW_SIZE
    WINDOW_SIZE = window  # optional param overrides default

    emotions = getEmotionsDataFrame()

    # Dataframes are indexed by data
    btc = pd.read_csv('timeseries/btc.csv', index_col='date')
    # dev = pd.read_csv('timeseries/dev.csv', index_col='date')
    share = pd.read_csv('timeseries/share.csv', index_col='date')

    dev = emotions[[]]
    for em in EMOTIONS:
        calculateStandardDev(dev, emotions, em)

    # Reconstruct datetime indecies from string
    # dev.index = pd.to_datetime(dev.index)
    btc.index = pd.to_datetime(btc.index)
    share.index = pd.to_datetime(share.index)

    btc.dropna(axis="rows", inplace=True)
    emotions.dropna(axis="rows", inplace=True)

    if plot is not None:
        render(emotions, share, dev, btc)

    if train is not None:

        test_data = dev.merge(btc, right_index=True, left_index=True, how="left")[
            EMOTIONS[:-1] + ['return', 'close', 'volume']]
        model = (trainXGBoost if train == 'xgboost' else trainLSTM)(test_data)


if __name__ == "__main__":
    """
      CLI APP example commands:
        1. python3 main.py --plot --window=24
        2. python3 main.py --plot --serve
        3. python3 main.py --train
    """
    fire.Fire(main)
