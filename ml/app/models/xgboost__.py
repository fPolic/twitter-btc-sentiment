import xgboost as xgb

import plotly.express as px
import matplotlib.pyplot as plt

from plotly.subplots import make_subplots
from utils.dataset_split import split_dataframe


def train(data):

    X_train, y_train, X_test, y_test = split_dataframe(data)

    reg = xgb.XGBRegressor(n_estimators=1000)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=50,
            verbose=True)
    xgb.plot_importance(reg, height=0.9)

    pred = reg.predict(X_test)
    X_test['pred'] = pred

    fig = make_subplots(rows=1, cols=1,  shared_xaxes=True)

    fig.add_scatter(
        x=X_test.index, y=y_test['sum_returns'], mode='lines', name='real')
    fig.add_scatter(
        x=X_test.index, y=X_test['pred'], mode='lines', name='predicted')

    fig.update_layout(title_text="Predicted vs. real", height=500)
    fig.show()
