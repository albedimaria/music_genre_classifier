import numpy as np
import tensorflow.keras as keras

from nn_genre_classifier import plot_history
from utils import config_loader
from sklearn.model_selection import train_test_split

DATASET_PATH = "D:\Projects\CNN_genre_classificator\dataset\Data\data.json"


def prepare_datasets(test_size, validation_size):
    # load data
    X, y = config_loader.load_data(DATASET_PATH)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # create train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    # create the RNN model

    # build network topology
    model = keras.Sequential()

    # 2 LSTM layers
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))

    # dense layer
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3)) # prevent overfitting by randomly turning off neurons

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model


if __name__ == '__main__':
    # create train, validation abd test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # build the CNN net
    input_shape = (X_train.shape[1], X_train.shape[2]) # 130, 13
    model = build_model(input_shape)

    # compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    # train the model
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=10)

    # plot loss and accuracy
    plot_history(history)

    # evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f'Test accuracy: {test_accuracy}')


