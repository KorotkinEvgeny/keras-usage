from __future__ import print_function
import numpy as np
import json
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from logger import get_logger

TRAIN_ROWS = 60000
TEST_ROWS = 10000


def create_layer():
    pass


def main():
    np.random.seed(1671)  # for reproducibility

    # network and training
    optimizer = SGD()  # SGD optimizer, explained later in this chapter

    # data: shuffled and split between train and test sets

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # X_train is 60000 rows of 28x28 values --> reshaped in 60000 x 784

    # TODO move it constants somewhere else (TO config maybe)
    X_train = X_train.reshape(TRAIN_ROWS, CONFIG.get('RESHAPED'))
    X_test = X_test.reshape(TEST_ROWS, CONFIG.get('RESHAPED'))
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # normalize
    #
    X_train /= 255
    X_test /= 255
    logger.info('%s train samples \n%s test samples', X_train.shape[0], X_test.shape[0])

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, CONFIG.get('NB_CLASSES'))
    Y_test = np_utils.to_categorical(y_test, CONFIG.get('NB_CLASSES'))

    # 10 outputs
    # final stage is softmax

    model = Sequential(
        [Dense(CONFIG.get('NB_CLASSES'), input_shape=(CONFIG.get('RESHAPED'),)),
         Activation('softmax')])

    # TODO could pass it in a constructor
    # EXAMPLE:
    # model = Sequential([
    #     Dense(32, input_shape=(784,)),
    #     Activation('relu'),
    #     Dense(10),
    #     Activation('softmax'),
    # ])

    # model.add(Dense(CONFIG.get('NB_CLASSES'), input_shape=(CONFIG.get('RESHAPED'),)))
    # model.add(Activation('softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train,
                        batch_size=CONFIG.get('BATCH_SIZE'),
                        epochs=CONFIG.get('NB_EPOCH'),
                        verbose=CONFIG.get('VERBOSE'),
                        validation_split=CONFIG.get('VALIDATION_SPLIT'))  # how much TRAIN is reserved for VALIDATION

    score = model.evaluate(X_test, Y_test, verbose=CONFIG.get('VERBOSE'))
    logger.info("\nTest score: %s \nTest accuracy: %s", score[0], score[1])


if __name__ == "__main__":
    logger = get_logger(__name__)

    with open('chapter1_config.json') as json_data:
        CONFIG = json.load(json_data)['V1']

    main()
