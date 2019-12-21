import datetime
import tensorflow as tf
import matplotlib.pyplot as plt

from numpy import add, arange, array, column_stack

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, LSTM, Activation, Dropout, Dense, Bidirectional

# This 3 params define input shape dimension
WINDOW_SIZE = 24
EMOTIONS_DIMENSIONS = 2
NUMBER_OF_DAYS = 90


def getDataInShape(data):

    fear = data['fear'].head(NUMBER_OF_DAYS * WINDOW_SIZE).values
    negative = data['negative'].head(NUMBER_OF_DAYS * WINDOW_SIZE).values

    btc_return = data['return'].head(NUMBER_OF_DAYS * WINDOW_SIZE).values

    X = array(column_stack((fear, negative))).reshape(
        NUMBER_OF_DAYS, WINDOW_SIZE, EMOTIONS_DIMENSIONS)

    # Y = btc_return[::NUMBER_OF_DAYS]
    Y = add.reduceat(btc_return, arange(0, len(btc_return), WINDOW_SIZE))

    return tf.convert_to_tensor(X, dtype=tf.float32), tf.convert_to_tensor(Y, dtype=tf.float32)


def bidirectional(data):

    X, Y = getDataInShape(data)

    X_TEST = tf.reshape(X[-1], (1, 24, 2))
    Y_TEST = Y[-1]

    X = X[:-1]
    Y = Y[:-1]

    log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(
        log_dir=log_dir, histogram_freq=1)

    model = Sequential([
        Bidirectional(LSTM(WINDOW_SIZE, activation='relu',
                           return_sequences=True), input_shape=(WINDOW_SIZE, EMOTIONS_DIMENSIONS)),
        Bidirectional(LSTM(WINDOW_SIZE, activation='relu')),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])
    model.fit(X, Y, epochs=1000, validation_split=0.2,
              batch_size=WINDOW_SIZE, callbacks=[tensorboard_callback])
    print(model.summary())

    baseline = data['return'].tail(
        WINDOW_SIZE * EMOTIONS_DIMENSIONS * NUMBER_OF_DAYS).mean()
    print(model.predict(X_TEST, verbose=True), Y_TEST, baseline)
    return model
