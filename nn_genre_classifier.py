import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from utils import config_loader
from tensorflow import keras
import matplotlib.pyplot as plt

DATASET_PATH = os.path.join(config_loader.get_dataset_path(), "data.json")


def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # convert a list into a numpy array
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    return X, y


def plot_history(history):
    fig, axs = plt.subplots(2)

    # create the accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("accuracy eval")

    # create the accuracy subplot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("error")
    axs[1].set_xlabel("epochs")
    axs[1].legend(loc="upper right")
    axs[1].set_title("error eval")

    plt.show()



if __name__ == "__main__":
    # check on TF
    # print("TensorFlow version:", tf.__version__)
    # print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

    # load data
    X, y = load_data(DATASET_PATH)

    # split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # build the net architecture
    model = keras.Sequential([
        # input layer
        keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),

        # 1st dense layer
        keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # 2st dense layer
        keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # 3st dense layer
        keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # output layer
        keras.layers.Dense(10, activation="softmax")
    ])

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train network
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=20,
                        batch_size=32)

    # plot accuracy and error as a function of the epochs
    plot_history(history)

    # make predictions on a sample
