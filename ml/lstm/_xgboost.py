import xgboost as xgb

import matplotlib.pyplot as plt

SPLIT_DATE = '2019-01-14'
print(xgb)


def trainModel(data):
    data.index = data.date
    data.sort_index(inplace=True)
    data = data.loc['2018-09-02':]

    data['sum_returns'] = data['return'].rolling(24).sum()
    data['sum_returns'] = data['sum_returns'].shift(-23)
    data.dropna(axis="rows", inplace=True)

    train = data.loc[data.date < SPLIT_DATE]
    test = data.loc[data.date > SPLIT_DATE]

    X_train = train[['close', 'positive', 'negative', 'anticipation', 'trust']]
    y_train = train[['sum_returns']]

    X_test = test[['close', 'positive', 'negative', 'anticipation', 'trust']]
    y_test = test[['sum_returns']]

    reg = xgb.XGBRegressor(n_estimators=1000)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=50,  # stop if 50 consequent rounds without decrease of error
            verbose=True)  # Change verbose to True if you want to see it train
    # xgb.plot_importance(reg, height=0.9)

    pred = reg.predict(X_test)
    plt.figure(figsize=(15, 3))
    plt.xlabel('time')
    plt.plot(X_test.index, pred, label='data')
    plt.plot(test.index, test['sum_returns'].shift(23), label='prediction')
    plt.show()
