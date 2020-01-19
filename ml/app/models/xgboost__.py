import xgboost as xgb

from numpy import around, arange, sqrt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from plotly import offline
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.dataset_split import split_dataframe

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


def train_linear(data, fig):
    X_train, y_train, X_test, y_test = split_dataframe(
        data, model_type="linreg")

    lin = LinearRegression()
    lin.fit(X_train, y_train['target'])

    pred_lin = lin.predict(X_test)

    lin_r2 = "Lin. reg. R2: " + \
        str(around(r2_score(y_test['target'], pred_lin), decimals=4))

    fig.add_scatter(x=X_test.index, y=pred_lin, mode='lines', marker_color=colors[2],
                    name='Linear reg. predicted', row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[X_test.index[100]],
        y=[0.075],
        mode='text',
        text=[lin_r2],
        name='Linear reg. R2',
        textposition='bottom right',
        textfont=dict({
             'size': 12,
        })
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

    xgb_r2 = "XGBoost R2: " + \
        str(around(r2_score(y_test['target'], pred), decimals=4))

    features = {k: v for k, v in sorted(
        reg.get_booster().get_score().items(), key=lambda item: item[1])}

    f = list()
    fv = list()
    for k, v in features.items():
        f.append(k)
        fv.append(v)

    print(features)
    print(xgb_r2)

    fig.add_scatter(
        x=X_test.index, y=y_test['target'], marker_color=colors[0], mode='lines', name='BTC absolute returns', row=1, col=1)
    fig.add_scatter(
        x=X_test.index, y=pred, mode='lines', marker_color=colors[1], name='XGBoost predicted', row=1, col=1)

    fig.add_trace(go.Bar(x=fv, y=f, orientation='h', name='XGBoost feature importance',
                         width=0.5, marker_color='#333'), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=[X_test.index[100]],
        y=[0.067],
        mode='text',
        text=[xgb_r2],
        name='XGBoost R2',
        textposition='bottom right',
        textfont=dict({
            'size': 12,
        })
    ))


def train(data):

    fig = make_subplots(shared_xaxes=False, specs=[
        [{"colspan": 2}, None], [{}, {}]], rows=2, cols=2)

    train_linear(data.copy(), fig)
    train_xgboost(data.copy(), fig)

    fig.update_layout(title_text="Model performance",
                      legend_orientation="v", height=900)
    offline.plot(fig, filename='static/model.html', auto_open=True)
