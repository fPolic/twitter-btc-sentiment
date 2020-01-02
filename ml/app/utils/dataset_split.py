
__SPLIT_DATE__ = '2019-01-14'
# Drop data before this date
FROM_DATA = '2018-09-02'

# data - -> Pandas dataframe indexed by date


def split_dataframe(data, SPLIT_DATE=__SPLIT_DATE__):
    data = data.loc[FROM_DATA:]

    data['sum_returns'] = data['return'].rolling(24).sum()
    data['sum_returns'] = data['sum_returns'].shift(-23)
    data.dropna(axis="rows", inplace=True)

    train = data.loc[data.index < SPLIT_DATE]
    test = data.loc[data.index > SPLIT_DATE]

    X_train = train[['close', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative',
                     'positive', 'sadness', 'surprise', 'trust']]
    y_train = train[['sum_returns']]

    X_test = test[['close', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative',
                   'positive', 'sadness', 'surprise', 'trust']]
    y_test = test[['sum_returns']]

    return X_train, y_train, X_test, y_test
