
__SPLIT_DATE__ = '2018-09-30'
# Drop data before this date
# FROM_DATA = '2018-09-02'


def process_df_for_linreg(data):
    data['return'] = data['return'].abs()
    data['volume'] = data['volume'].diff()
    data['hour'] = data.index.hour

    data['target'] = data['return'].shift(-1)

    # data['dayOfWeek'] = data.index.dayofweek
    # data['dayOfMonth'] = data.index.day

    data.dropna(axis="rows", inplace=True)


def process_df_for_xgboost(data):
    data['return'] = data['return'].abs()
    data['volume'] = data['volume']
    data['hour'] = data.index.hour

    data['target'] = data['return'].shift(-1)

    # data['dayOfWeek'] = data.index.dayofweek
    # data['dayOfMonth'] = data.index.day

    data.fillna(0, inplace=True)


def split_dataframe(data, model_type="xgboost", SPLIT_DATE=__SPLIT_DATE__):
    # data -> Pandas dataframe indexed by date

    # data = data.loc[FROM_DATA:]

    if model_type == 'linreg':
        process_df_for_linreg(data)
    else:
        process_df_for_xgboost(data)

    train = data.loc[data.index < SPLIT_DATE]
    test = data.loc[data.index > SPLIT_DATE]

    FEATURES = ['return',  'volume',  'anticipation', 'sadness', 'hour']
    TARGET = ['target']

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    return X_train, y_train, X_test, y_test
