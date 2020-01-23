import fire
import math
import datetime
import pandas as pd


from numpy import array
from pymongo import MongoClient, DESCENDING

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

colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]


def getEmotionsDataFrame():
    collection = DB["emotions_aggregation"]
    cursor = list(collection.find(no_cursor_timeout=True))

    _emotion = pd.DataFrame(cursor)
    _emotion.set_index("date", inplace=True)
    _emotion.index = pd.to_datetime(_emotion.index)

    return _emotion[EMOTIONS]


def getTopWords(field="count"):
    collection = DB["word_count"]

    params = dict()
    params['_id'] = 1
    params[field] = 1

    data = collection.find({}, params, no_cursor_timeout=True).sort(
        field, DESCENDING).limit(10)
    cursor = list(data)

    return pd.DataFrame(cursor)


def getSPX():
    collection = DB["spx"]
    cursor = list(collection.find(no_cursor_timeout=True))

    spx = pd.DataFrame(cursor)
    spx.set_index("date", inplace=True)
    spx.index = pd.to_datetime(spx.index)

    spx['return'] = (spx['close'] / spx['open']) - 1

    return spx[['close', 'volume', 'return']].rename(columns={'close': 'spx_close',
                                                              'volume': 'spx_volume', 'return': 'spx_return'})


def calculateStandardDev(dev, df, column):
    dev[column] = (df[column]-df[column].mean())/df[column].std(ddof=0)

# =================== PLOTING TIME SERIES ===================


def render(emotions, share, dev, btc):

    # ======================================== WORD COUNT ========================================

    fig = make_subplots(
        rows=4, cols=3, subplot_titles=EMOTIONS,  vertical_spacing=0.1)

    for ind, em in enumerate(EMOTIONS):
        words = getTopWords(em)
        fig.add_trace(
            go.Bar(y=words['_id'], x=words[em], name=em, orientation='h'), row=math.floor(ind / 3) + 1, col=ind % 3 + 1)

    fig.update_layout(title_text="Word count by emotion", height=1200, yaxis=dict(
        dtick=1, autorange="reversed"
    ),
        yaxis2=dict(
        dtick=1, autorange="reversed"
    ),
        yaxis3=dict(
        dtick=1, autorange="reversed"
    ),
        yaxis4=dict(
        dtick=1, autorange="reversed"
    ),
        yaxis5=dict(
        dtick=1, autorange="reversed"
    ),
        yaxis6=dict(
        dtick=1, autorange="reversed"
    ),
        yaxis7=dict(
        dtick=1, autorange="reversed"
    ),
        yaxis8=dict(
        dtick=1, autorange="reversed"
    ),
        yaxis9=dict(
        dtick=1, autorange="reversed"
    ),
        yaxis10=dict(
        dtick=1, autorange="reversed"
    ),
        yaxis11=dict(
        dtick=1, autorange="reversed"
    ))
    offline.plot(fig, filename='static/words.html', auto_open=True)

    # ======================================== BOXPLOTS (H) ========================================

    fig = make_subplots(rows=5, cols=2, subplot_titles=[
        x for x in EMOTIONS[:-1]])

    row = 0
    groups = emotions.groupby([emotions.index.hour])
    for em in EMOTIONS[:-1]:
        row = row + 1
        for key, group in groups:
            fig.add_trace(go.Box(
                y=group[em].values,
                name=str(key),
                line=dict(color=colors[row-1])
            ), row=row % 5 + 1, col=row % 2 + 1)

        avg = emotions[em].mean()
        fig.add_scatter(y=[avg for _ in range(
            24)], mode="lines+markers", line=dict(width=4), row=row % 5 + 1, col=row % 2 + 1)

    fig.update_layout(
        title_text="Emotion box plots hourly", showlegend=False, height=1500)

    offline.plot(fig, filename='static/boxplots-hourly.html',
                 auto_open=True)

    # ======================================== BOXPLOTS (D) ========================================

    fig = make_subplots(rows=5, cols=2, subplot_titles=[
        x for x in EMOTIONS[:-1]])

    row = 0
    days = ['Monday', 'Tuesday', 'Wednesday',
            'Thursday', 'Friday', 'Saturday', 'Sunday']
    by_day = dict()
    for key, group in emotions.groupby([emotions.index.weekday_name]):
        by_day[key] = group
    for em in EMOTIONS[: -1]:
        row = row + 1
        for day in days:
            fig.add_trace(go.Box(
                y=by_day[day][em].values,
                name=str(day),
                line=dict(color=colors[row-1])
            ), row=row % 5 + 1, col=row % 2 + 1)

        avg = emotions[em].mean()
        fig.add_scatter(y=[avg for _ in range(
            7)], x=days, mode="lines+markers", line=dict(width=4), row=row % 5 + 1, col=row % 2 + 1)

    fig.update_layout(
        title_text="Emotion box plots daily", showlegend=False, height=1500)

    offline.plot(fig, filename='static/boxplots-daily.html',
                 auto_open=True)

    # ======================================== BOXPLOTS (M) ========================================

    fig = make_subplots(rows=5, cols=2, subplot_titles=[
                        x for x in EMOTIONS[:-1]])

    row = 0
    groups = emotions.groupby([emotions.index.month])
    for em in EMOTIONS[:-1]:
        row = row + 1
        for key, group in groups:
            fig.add_trace(go.Box(
                y=group[em].values,
                name=str(key),
                line=dict(color=colors[row-1])
            ), row=row % 5 + 1, col=row % 2 + 1)

        avg = emotions[em].mean()
        fig.add_scatter(y=[avg for _ in range(
            12)], x=list(range(1, 13)), mode="lines+markers", line=dict(width=4), row=row % 5 + 1, col=row % 2 + 1)

    fig.update_layout(
        title_text="Emotion box plots monthly", showlegend=False, height=1500)

    offline.plot(fig, filename='static/boxplots-monthly.html',
                 auto_open=True)

    # ======================================== OVERVIEW ========================================

    fig = make_subplots(rows=3, cols=2, shared_xaxes=True, specs=[
        [{"colspan": 2, }, None],
        [{}, {}],
        [{"colspan": 2, }, None]
    ], subplot_titles=(
        'Number of words associated with emotion (per hour)', 'Global boxplot for emotion', 'Total word count per emotion', 'Bitcoin close price'))

    i = -1
    sums = dict()

    for em in EMOTIONS[:-1]:
        i += 1
        fig.add_scatter(x=emotions.index, y=emotions[em], legendgroup=em,
                        mode='lines', row=1, col=1, name=em, line=dict(color=colors[i]))
        fig.add_box(y=emotions[em], legendgroup=em,
                    row=2, col=1, name="Box/" + em, line=dict(color='#fecb52'))

        sums[em] = emotions[em].sum()

    sums = list(sums.items())
    sums.sort(key=lambda x: x[1])

    fig.add_trace(go.Bar(y=[i[0] for i in sums], x=[i[1] for i in sums], orientation='h',
                         name='Total word counts per emotion', marker_color='#636dfa'), row=2, col=2)

    btc = btc.merge(emotions[[]], right_index=True,
                    left_index=True, how='right')

    fig.add_scatter(x=btc.index, y=btc['close'],  col=1,
                    mode='lines', row=3, name='Bitcoin close price (hourly)')

    fig.update_layout(title_text="Overview")
    offline.plot(fig, filename='static/overview.html', auto_open=True)

    # ======================================== EMOTIONS ========================================

    fig = make_subplots(rows=9, cols=2, specs=[
        [{"colspan": 2, 'rowspan': 2}, None],
        [{}, {}],
        [{}, {}],
        [{}, {}],
        [{}, {}],
        [{}, {}],
        [{}, {}],
        [{"rowspan": 2}, {"rowspan": 2}],
        [{}, {}]], subplot_titles=(
        'Share of words containing certain emotion in total word count', '', '', *EMOTIONS[: -1]))

    i = -1
    for em in EMOTIONS[: -1]:
        i += 1
        fig.add_scatter(x=share.index, y=share[em],
                        mode='lines', row=1, col=1, name="Share/" + em, legendgroup=em, line=dict(color=colors[i]))

        fig.add_scatter(x=dev.index, y=dev[em],
                        mode='lines', row=3 + (i % 5), col=(i % 2) + 1, name="Standard dev./" + em, legendgroup=em, line=dict(color=colors[i]))

    fig.update_layout(title_text="Emotion share", height=1000)
    offline.plot(fig, filename='static/emotions.html', auto_open=True)

    # ======================================== DISTRIBUTION ========================================

    fig = make_subplots(rows=3,  shared_xaxes=False,
                        subplot_titles=('Emotions histogram'))

    fig = ff.create_distplot(
        [share[c] for c in EMOTIONS[:-1]], EMOTIONS[:-1], bin_size=0.0)

    fig.update_layout(
        title_text="Emotion distribution histogram", legend_orientation="h")

    offline.plot(fig, filename='static/histogram.html', auto_open=True)


# =================== MAIN CLI APP ===================


def main(plot=None, train=None, serve=False, window=WINDOW_SIZE):
    global WINDOW_SIZE
    WINDOW_SIZE = window  # optional param overrides default

    emotions = getEmotionsDataFrame()
    # spx = getSPX()

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
        # test_data = test_data.merge(
        #     spx, right_index=True, left_index=True, how="left")
        model = (trainXGBoost if train == 'xgboost' else trainLSTM)(test_data)


if __name__ == "__main__":
    """
      CLI APP example commands:
        1. python3 main.py --plot --window=24
        2. python3 main.py --plot --serve
        3. python3 main.py --train
    """
    fire.Fire(main)
