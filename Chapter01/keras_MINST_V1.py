from __future__ import print_function

from collections import namedtuple

import fire
import numpy as np
import json

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from logger import get_logger
from KerasBaseModel import KerasBaseModel

TRAIN_ROWS = 60000
TEST_ROWS = 10000

np.random.seed(1671)  # for reproducibility
Dataset = namedtuple('Dataset', ['X_train', 'Y_train', 'X_test', 'Y_test'])
Score = namedtuple('Score', ['test_score', 'test_accuracy'])

class KerasNetworkCh1(KerasBaseModel):

    def __init__(self):
        self._model = None
        self._score = None
        self._dataset = None

    @property
    def dataset(self):
        if not self._dataset:
            (X_train, y_train), (X_test, y_test) = mnist.load_data()

            # X_train is 60000 rows of 28x28 values --> reshaped in 60000 x 784

            X_train = X_train.reshape(TRAIN_ROWS, self.reshaped) / 255
            X_test = X_test.reshape(TEST_ROWS, self.reshaped) / 255
            X_train = X_train.astype('float32')
            X_test = X_test.astype('float32')
            Y_train = np_utils.to_categorical(y_train, self.nb_classes)
            Y_test = np_utils.to_categorical(y_test, self.nb_classes)
            self._dataset = Dataset(X_train, Y_train, X_test, Y_test)
        return self._dataset

    #todo move it to method and call when model set up
    @property
    def model(self):
        if not self._model:
            self._model = Sequential(
                [Dense(self.nb_classes,
                       input_shape=(self.reshaped,)),
                 Activation('softmax')])
            self._model.summary()

            self._model.compile(loss='categorical_crossentropy',
                                optimizer=self.optimizer,
                                metrics=['accuracy'])

            history = self._model.fit(self.dataset.X_train, self.dataset.Y_train,
                                      batch_size=self.batch_size,
                                      epochs=self.nb_epoch,
                                      verbose=self.verbose,
                                      validation_split=self.validation_split)
        return self._model

    @property
    def score(self):
        if not self._score:
            self._score = self.model.evaluate(
                self.dataset.X_test,
                self.dataset.Y_test,
                verbose=self.verbose
            )

        return Score(self._score[0], self._score[1])


def main():

    # keras1 = KerasNetworkCh1()
    fire.Fire(KerasNetworkCh1)

    # print(keras1.dataset.X_train.shape[0], 'train samples')
    # print(keras1.dataset.X_test.shape[0], 'test samples')
    # print("\nTest score:", keras1.score.test_score)
    # print('Test accuracy:', keras1.score.test_accuracy)


if __name__ == "__main__":
    logger = get_logger(__name__)

    with open('chapter1_config.json') as json_data:
        CONFIG = json.load(json_data)['V1']

    main()
