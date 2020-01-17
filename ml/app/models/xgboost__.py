import xgboost as xgb

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

from numpy import around
from plotly import offline
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from matplotlib import pyplot
from utils.dataset_split import split_dataframe


def train(data):

    X_train, y_train, X_test, y_test = split_dataframe(data)

    lin = LinearRegression()
    lin.fit(X_train, y_train['target'])

    reg = xgb.XGBRegressor(n_estimators=1000)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=50,
            verbose=False)

    pred = reg.predict(X_test)
    pred_lin = lin.predict(X_test)
    print(X_train.shape, y_test.shape)

    X_test['pred'] = pred

    xgb_r2 = "XGBoost r2: " + \
        str(around(r2_score(y_test['target'], pred), decimals=4))
    lin_r2 = "Lin. reg. r2: " + \
        str(around(r2_score(y_test['target'], pred_lin), decimals=4))

    print("XGBoost feature importance: ", reg.get_booster().get_score())
    print(xgb)
    print(lin_r2)

    # ==================== PLOT RESULTS & FEATURES IMPORTANCE ===================

    fig = make_subplots(shared_xaxes=False, rows=2, cols=1)
    fig.add_scatter(
        x=X_test.index, y=y_test['target'], marker_color=y_test['target'], mode='lines', name='BTC absolute returns', row=1, col=1)
    fig.add_scatter(
        x=X_test.index, y=X_test['pred'], mode='lines', name='XGBoost predicted', row=1, col=1)
    fig.add_scatter(x=X_test.index, y=pred_lin, mode='lines',
                    name='Linear reg. predicted', row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[X_test.index[150], X_test.index[150]],
        y=[0.077, 0.067],
        mode='text',
        name='R2 score',
        text=[lin_r2, xgb_r2],
        textposition='bottom center'
    ))

    fig.update_layout(title_text="Model performances",
                      legend_orientation="v", height=900)
    offline.plot(fig, filename='static/model.html', auto_open=True)
