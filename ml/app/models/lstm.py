import datetime
import operator
import functools

import tensorflow as tf
import matplotlib.pyplot as plt

from numpy import add, arange, array, column_stack, float32

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, LSTM, Activation, Dropout, Dense, Bidirectional

from utils.dataset_split import split_dataframe

# This 3 params define input shape dimension
WINDOW_SIZE = 1
EMOTIONS_DIMENSIONS = 11
NUMBER_OF_DAYS = WINDOW_SIZE * 30


def getDataInShape(data):
    X_train, y_train, X_test, y_test = split_dataframe(data)

    X_train.head(NUMBER_OF_DAYS * WINDOW_SIZE)

    anger = X_train['anger']
    anticipation = X_train['anticipation']
    disgust = X_train['disgust']
    fear = X_train['fear']
    joy = X_train['joy']
    negative = X_train['negative']
    positive = X_train['positive']
    sadness = X_train['sadness']
    surprise = X_train['surprise']
    trust = X_train['trust']
    return_ = X_train['return']

    Y = y_train['target'].values
    X = array(column_stack(
        (anger, anticipation, disgust, fear, joy, negative, positive, sadness, surprise, trust, return_)))

    X = X.reshape(
        (X.shape[0], X.shape[1], 1))

    return tf.convert_to_tensor(X, dtype=tf.float32), tf.convert_to_tensor(Y, dtype=tf.float32), X_test, y_test['target'].values


def bidirectional():
    return Sequential([
        Bidirectional(LSTM(EMOTIONS_DIMENSIONS, activation='relu',
                           return_sequences=True), input_shape=(EMOTIONS_DIMENSIONS, WINDOW_SIZE)),
        # Dropout(0.3),
        Bidirectional(LSTM(EMOTIONS_DIMENSIONS, activation='relu')),
        Dropout(0.2),
        Dense(1)
    ])


def stacked():
    return Sequential([
        LSTM(200, activation='relu',
             input_shape=(WINDOW_SIZE, EMOTIONS_DIMENSIONS)),
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
    # plt.title(
    #     'model train vs validation loss, [PREDICTED]: ' + tf.strings.as_string(predict) + ' [REAL]: ' + tf.strings.as_string(real))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()


def train(data, lstm_type='bidirectional', plot_loss=True):

    X, Y, X_TEST, Y_TEST = getDataInShape(data)

    log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(
        log_dir=log_dir, histogram_freq=1)

    model = stacked() if lstm_type == 'stacked' else bidirectional()
    model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])

    history = model.fit(X, Y, epochs=100, validation_split=0.2,
                        batch_size=WINDOW_SIZE * 24, callbacks=[tensorboard_callback])

    if (plot_loss):
        plotLoss(history)

    for i in range(0, len(X_TEST)):
        x = model.predict(tf.reshape(
            X_TEST[i], (1, WINDOW_SIZE, EMOTIONS_DIMENSIONS)), verbose=True)
        y = Y_TEST[i]
        print(x, y)

    return model
