import pandas as pd


def process_df_for_linreg(data):
    data['return'] = data['return'].abs()
    data['volume'] = data['volume'].diff()
    data['hour'] = data.index.hour

    data['target'] = data['return'].shift(-1)
    # data['spx_return'] = data['spx_return'].abs()

    # data['dayOfWeek'] = data.index.dayofweek
    # data['dayOfMonth'] = data.index.day

    data.dropna(axis="rows", inplace=True)


def process_df_for_xgboost(data):
    data['return'] = data['return'].abs()
    data['volume'] = data['volume'].diff()
    data['hour'] = data.index.hour

    # data['spx_return'] = data['spx_return'].abs()

    data['target'] = data['return'].shift(-1)

    # data['dayOfWeek'] = data.index.dayofweek
    # data['dayOfMonth'] = data.index.day

    data.fillna(0, inplace=True)


EMOTIONS = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative',
            'positive', 'sadness', 'surprise', 'trust', 'count']
# data -> Pandas dataframe indexed by date


def split_dataframe(data, model_type="xgboost", SPLIT_DATE='2018-09-30'):

    if model_type == 'linreg':
        process_df_for_linreg(data)
    else:
        process_df_for_xgboost(data)

    data = data.round(4)

    # data['dd'] = data.index
    # data['gap'] = (data['dd'].diff() /
    #                pd.to_timedelta('1 hour')).fillna(0)

    # print(data.head(200))
    # data = data[data['gap'] < 2]

    train = data.loc[data.index < SPLIT_DATE]
    test = data.loc[data.index > SPLIT_DATE]

    FEATURES = ['return', 'volume', 'sadness',
                'negative', 'anticipation', 'disgust', 'hour']

    TARGET = ['target']

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    print("------> mean:", test[TARGET].mean())
    print(test[TARGET].std(ddof=0))

    return X_train, y_train, X_test, y_test
