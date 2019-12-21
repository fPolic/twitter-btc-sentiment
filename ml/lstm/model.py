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
NUMBER_OF_DAYS = 210


def getDataInShape(data):

    fear = data['fear'].head(NUMBER_OF_DAYS * WINDOW_SIZE).values
    count = data['count'].head(NUMBER_OF_DAYS * WINDOW_SIZE).values

    btc_return = data['return'].head(NUMBER_OF_DAYS * WINDOW_SIZE)
    _btc_return = ((btc_return - btc_return.min()) / (btc_return.max() -
                                                      btc_return.min())).values

    # X = []
    # Y = []
    # cols = array(column_stack((fear, count)))
    # for i in range(0, cols.shape[0] - WINDOW_SIZE):
    #     X.append(cols[i:i + WINDOW_SIZE])
    #     Y.append(data['close'].values[(i + 1) * WINDOW_SIZE])

    X = array(column_stack((fear, count))).reshape(
        NUMBER_OF_DAYS, WINDOW_SIZE, EMOTIONS_DIMENSIONS)
    Y = add.reduceat(_btc_return, arange(0, _btc_return.size, WINDOW_SIZE))

    return tf.convert_to_tensor(X, dtype=tf.float32), tf.convert_to_tensor(Y, dtype=tf.float32)


def bidirectional(data):

    X, Y = getDataInShape(data)

    X_TEST = tf.reshape(X[-1], (1, WINDOW_SIZE, EMOTIONS_DIMENSIONS))
    Y_TEST = Y[-1]

    X = X[:-1]
    Y = Y[:-1]

    log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(
        log_dir=log_dir, histogram_freq=1)

    model = Sequential([
        Bidirectional(LSTM(WINDOW_SIZE, activation='relu',
                           return_sequences=True), input_shape=(WINDOW_SIZE, EMOTIONS_DIMENSIONS)),
        Dropout(0.4),
        Bidirectional(LSTM(WINDOW_SIZE, activation='relu')),
        Dense(1)
    ])

    # model = Sequential()
    # model.add(LSTM(200, activation='relu',
    #                return_sequences=True, input_shape=(WINDOW_SIZE, 2)))
    # model.add(LSTM(100, activation='relu', return_sequences=True))
    # model.add(LSTM(50, activation='relu', return_sequences=True))
    # model.add(LSTM(25, activation='relu'))
    # model.add(Dense(20, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(10, activation='relu'))
    # model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=[
                  'mean_squared_error'])
    history = model.fit(X, Y, epochs=500, validation_split=0.2,
                        batch_size=WINDOW_SIZE, callbacks=[tensorboard_callback])

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

    baseline = data['return'].tail(
        WINDOW_SIZE * EMOTIONS_DIMENSIONS * NUMBER_OF_DAYS).mean()
    print(model.predict(X_TEST, verbose=True), Y_TEST, baseline)
    return model
