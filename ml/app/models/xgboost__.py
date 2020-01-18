import xgboost as xgb

from numpy import around, arange, sqrt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from plotly import offline
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.dataset_split import split_dataframe


def train_linear(data, fig):
    X_train, y_train, X_test, y_test = split_dataframe(
        data, model_type="linreg")

    lin = LinearRegression()
    lin.fit(X_train, y_train['target'])

    pred_lin = lin.predict(X_test)

    lin_r2 = "Lin. reg. R2: " + \
        str(around(r2_score(y_test['target'], pred_lin), decimals=4))

    fig.add_scatter(x=X_test.index, y=pred_lin, mode='lines',
                    name='Linear reg. predicted', row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[X_test.index[100]],
        y=[0.075],
        mode='text',
        text=[lin_r2],
        name='Linear reg. R2',
        textposition='bottom right'
    ))

    print(lin_r2)


def train_xgboost(data, fig):
    X_train, y_train, X_test, y_test = split_dataframe(data)

    reg = xgb.XGBRegressor(
        max_depth=14,
        learning_rate=0.2,
        n_estimators=1000,
        min_child_weight=300,
        colsample_bytree=0.8,
        subsample=0.8,
        eta=0.15,
        seed=50)

    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=10, verbose=False)

    pred = reg.predict(X_test)

    print(X_train.shape, y_test.shape)

    X_test['pred'] = pred

    xgb_r2 = "XGBoost R2: " + \
        str(around(r2_score(y_test['target'], pred), decimals=4))

    print("XGBoost feature importance: ", reg.get_booster().get_score())
    print(xgb_r2)

    fig.add_scatter(
        x=X_test.index, y=y_test['target'], marker_color=y_test['target'], mode='lines', name='BTC absolute returns', row=1, col=1)
    fig.add_scatter(
        x=X_test.index, y=X_test['pred'], mode='lines', name='XGBoost predicted', row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[X_test.index[100]],
        y=[0.067],
        mode='text',
        text=[xgb_r2],
        name='XGBoost R2',
        textposition='bottom right'
    ))


def train(data):

    fig = make_subplots(shared_xaxes=False, rows=2, cols=1)

    train_linear(data.copy(), fig)
    train_xgboost(data.copy(), fig)

    fig.update_layout(title_text="Model performances",
                      legend_orientation="v", height=900)
    offline.plot(fig, filename='static/model.html', auto_open=True)
