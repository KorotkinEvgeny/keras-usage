from __future__ import print_function
import json
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from logger import get_logger


def main():
    np.random.seed(1671)  # for reproducibility

    # network and training
    OPTIMIZER = SGD()  # optimizer, explained later in this chapter

    # data: shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # X_train is 60000 rows of 28x28 values --> reshaped in 60000 x 784

    X_train = X_train.reshape(60000, CONFIG.get("RESHAPED"))
    X_test = X_test.reshape(10000, CONFIG.get("RESHAPED"))
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # normalize
    X_train /= 255
    X_test /= 255
    # print(X_train.shape[0], 'train samples')
    # print(X_test.shape[0], 'test samples')

    logger.info('%s train samples \n%s test samples', X_train.shape[0], X_test.shape[0])

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, CONFIG.get('NB_CLASSES'))
    Y_test = np_utils.to_categorical(y_test, CONFIG.get('NB_CLASSES'))

    # M_HIDDEN hidden layers
    # 10 outputs
    # final stage is softmax

    model = Sequential()
    model.add(Dense(CONFIG.get('N_HIDDEN'), input_shape=(CONFIG.get('RESHAPED'),)))
    model.add(Activation('relu'))
    model.add(Dense(CONFIG.get('N_HIDDEN')))
    model.add(Activation('relu'))
    model.add(Dense(CONFIG.get('NB_CLASSES')))
    model.add(Activation('softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=OPTIMIZER,
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train,
                        batch_size=CONFIG.get('BATCH_SIZE'), epochs=CONFIG.get('NB_EPOCH'),
                        verbose=CONFIG.get('VERBOSE'), validation_split=CONFIG.get('VALIDATION_SPLIT'))

    score = model.evaluate(X_test, Y_test, verbose=CONFIG.get('VERBOSE'))
    logger.info("\nTest score: %s \nTest accuracy: %s", score[0], score[1])


if __name__ == '__main__':

    logger = get_logger(__name__)

    with open('chapter1_config.json') as json_data:
        CONFIG = json.load(json_data)['V2']

    main()
