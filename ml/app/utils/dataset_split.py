
__SPLIT_DATE__ = '2019-01-14'
# Drop data before this date
FROM_DATA = '2018-09-02'


def split_dataframe(data, SPLIT_DATE=__SPLIT_DATE__):
    # data -> Pandas dataframe indexed by date
    data = data.loc[FROM_DATA:]

    data['return'] = data['return'].abs()
    data['target'] = data['return'].shift(-1)

    data.dropna(axis="rows", inplace=True)

    train = data.loc[data.index < SPLIT_DATE]
    test = data.loc[data.index > SPLIT_DATE]

    FEATURES = ['return', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative',
                'positive', 'sadness', 'surprise']

    TARGET = ['target']

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    return X_train, y_train, X_test, y_test
