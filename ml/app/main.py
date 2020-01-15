import fire
import datetime
import pandas as pd

from numpy import array
from plotly import offline
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models.lstm import train as trainLSTM
from models.xgboost__ import train as trainXGBoost

WINDOW_SIZE = 24
EMOTIONS = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative',
            'positive', 'sadness', 'surprise', 'trust', 'count']

# =================== PLOTING TIME SERIES ===================


def render(share, dev, btc):
    btc = btc.merge(dev[['count']], right_index=True,
                    left_index=True, how='right')

    fig = make_subplots(rows=2, cols=2,  shared_xaxes=False, subplot_titles=(
        'Rolling z-score for 24h window', 'Emotion word count share in total count', 'Bitcoin price', 'Emotions boxplot'))

    # ==================== DEVIATIONS ====================
    for em in EMOTIONS:
        fig.add_scatter(x=dev.index, y=dev[em],
                        mode='lines', row=1, col=1, name="Standard dev./" + em)

    # ==================== SHARES ====================
    for em in EMOTIONS[:-1]:
        fig.add_scatter(x=share.index, y=share[em],
                        mode='lines', row=1, col=2, name="Share/" + em)
        fig.add_box(y=share[em],
                    row=2, col=2, name="Box/" + em)

    # ==================== BITCOIN ====================
    fig.add_scatter(x=btc.index, y=btc['close'],
                    mode='lines', row=2, col=1, name='Bitcoin close price (hourly)')

    # fig.add_bar(x=btc.index, y=btc['volume'],
    #             row=2, col=2, name='Bitcoin volume (hourly)')

    fig.update_layout(title_text="BTC tweets lexicon analysis")

    offline.plot(fig, filename='static/graphs.html', auto_open=True)

# =================== MAIN CLI APP ===================


def main(plot=None, train=None, serve=False, window=WINDOW_SIZE):
    global WINDOW_SIZE
    WINDOW_SIZE = window  # optional param overrides default

    # Dataframes are indexed by data
    btc = pd.read_csv('timeseries/btc.csv', index_col='date')
    dev = pd.read_csv('timeseries/dev.csv', index_col='date')
    share = pd.read_csv('timeseries/share.csv', index_col='date')

    if plot is not None:
        render(share, dev, btc)
        print(share.describe())

    if train is not None:

        test_data = share.merge(btc, right_index=True, left_index=True, how="inner")[
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
