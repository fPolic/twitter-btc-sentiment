import datetime
import operator
import functools

import tensorflow as tf
import matplotlib.pyplot as plt

from numpy import add, arange, array, column_stack

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, LSTM, Activation, Dropout, Dense, Bidirectional

# This 3 params define input shape dimension
WINDOW_SIZE = 23
EMOTIONS_DIMENSIONS = 2
NUMBER_OF_DAYS = 14


def getDataInShape(data):

    fear = data['fear'].head(NUMBER_OF_DAYS * WINDOW_SIZE).values
    count = data['count'].head(NUMBER_OF_DAYS * WINDOW_SIZE).values

    btc_return = data['return'].head(
        NUMBER_OF_DAYS * WINDOW_SIZE + WINDOW_SIZE + 1).values[WINDOW_SIZE:]
    # _btc_return = ((btc_return - btc_return.min()) / (btc_return.max() -
    #                                                   btc_return.min())).values

    X = []
    Y = []
    cols = array(column_stack((fear, count)))
    for i in range(0, cols.shape[0] - WINDOW_SIZE):
        X.append(cols[i:i + WINDOW_SIZE])
        Y.append(btc_return[i + WINDOW_SIZE])

    # X = array(column_stack((fear, count))).reshape(
    #     NUMBER_OF_DAYS, WINDOW_SIZE, EMOTIONS_DIMENSIONS)
    # Y = add.reduceat(btc_return, arange(0, btc_return.size, WINDOW_SIZE))

    return tf.convert_to_tensor(X, dtype=tf.float32), tf.convert_to_tensor(Y, dtype=tf.float32)


def bidirectional():
    return Sequential([
        Bidirectional(LSTM(WINDOW_SIZE, activation='relu',
                           return_sequences=True), input_shape=(WINDOW_SIZE, EMOTIONS_DIMENSIONS)),
        Dropout(0.3),
        Bidirectional(LSTM(WINDOW_SIZE, activation='relu')),
        Dense(1)
    ])


def stacked():
    return Sequential([
        LSTM(200, activation='relu',
             input_shape=(WINDOW_SIZE, 2)),
        LSTM(100, activation='relu', return_sequences=True),
        LSTM(50, activation='relu', return_sequences=True),
        LSTM(25, activation='relu'),
        Dense(20, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='relu'),
        Dense(1),
    ])


def plotLoss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()


def train(data, lstm_type='bidirectional', plot_loss=True):

    X, Y = getDataInShape(data)

    X_TEST = tf.reshape(X[-1], (1, WINDOW_SIZE, EMOTIONS_DIMENSIONS))
    Y_TEST = Y[-1]

    X = X[:-1]
    Y = Y[:-1]

    log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(
        log_dir=log_dir, histogram_freq=1)

    model = stacked() if lstm_type == 'stacked' else bidirectional()
    model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])
    history = model.fit(X, Y, epochs=250, validation_split=0.2,
                        batch_size=WINDOW_SIZE * 14, callbacks=[tensorboard_callback])

    if (plot_loss):
        plotLoss(history)

    print(model.predict(X_TEST, verbose=True), Y_TEST)
    return model
