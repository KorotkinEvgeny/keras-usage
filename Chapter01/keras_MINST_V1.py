from __future__ import print_function
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


class KerasNetwork(KerasBaseModel):


    def run_model(self):
        print(self.nb_epoch)

    # nb_epoch = 200
    # batch_size = 128
    # verbose_value = 1
    # nb_classes = 10
    # n_hidden = 128
    # validation_split = 0.2
    # reshaped = 784
    #
    # _optimizer = SGD()  # SGD optimizer, explained later in this chapter
    #
    # def __init__(self):
    #     self._model = None
    #     self._score = None
    #     self._dataset = None
    #     self.history = None
    #
    # @property
    # def model_epoch(self):
    #     return self.nb_epoch
    #
    # @model_epoch.setter
    # def model_epoch(self, epoch_input):
    #     self.nb_epoch = epoch_input
    #
    # @property
    # def verbose(self):
    #     return self.verbose_value
    #
    # @verbose.setter
    # def verbose(self, verbose_input):
    #     verbose_allowed_values = [1, 2, 3]
    #     if verbose_input in verbose_allowed_values:
    #         self.verbose_value = verbose_input
    #
    # @property
    # def model_description(self):
    #
    #     description = "Keras chapter 1 model " \
    #                   "\n Epoch: {0}" \
    #                   "\n Batch size: {1}" \
    #                   "\n Verbose {2}" \
    #                   "\n Classes {3}" \
    #                   "\n Hidden {4}" \
    #                   "\n Validation split {5}" \
    #                   "\n Reshaped {6}".format(
    #         self.nb_epoch,
    #         self.batch_size,
    #         self.verbose_value,
    #         self.nb_classes,
    #         self.n_hidden,
    #         self.validation_split,
    #         self.reshaped)
    #
    #     return description
    #
    # @property
    # def optimizer(self):
    #     return self._optimizer
    #
    # @optimizer.setter
    # def optimizer(self, input_optimizer):
    #     #TODO add Adam to import
    #     allowed_optimizers = ['SGD', 'Adam']
    #     if input_optimizer in allowed_optimizers:
    #         self._optimizer = input_optimizer
    #
    #TODO move it out of property
    @property
    def dataset(self):
        if not self.dataset:
            (X_train, y_train), (X_test, y_test) = mnist.load_data()

            # X_train is 60000 rows of 28x28 values --> reshaped in 60000 x 784

            X_train = X_train.reshape(TRAIN_ROWS, self.reshaped) / 255
            X_test = X_test.reshape(TEST_ROWS, self.reshaped) / 255
            X_train = X_train.astype('float32')
            X_test = X_test.astype('float32')
            Y_train = np_utils.to_categorical(y_train, self.nb_classes)
            Y_test = np_utils.to_categorical(y_test, self.nb_classes)

        return self.dataset


    #TODO Same as another property. Better create configuration model logic, and then compile and run. In it's case we could use existiong model, or change parameters and retrain it
    @property
    def model(self):
        if not self.model:
            self._model = Sequential(
                [Dense(self.nb_classes,
                       input_shape=(self.reshaped,)),
                 Activation('softmax')])
            self._model.summary()

            self._model.compile(loss='categorical_crossentropy',
                                optimizer=self._optimizer,
                                metrics=['accuracy'])

            history = self._model.fit(self.dataset.X_train, self.dataset.Y_train,
                                      batch_size=self.batch_size,
                                      epochs=self.nb_epoch,
                                      verbose=self.verbose_value,
                                      validation_split=self.validation_split)
        return self._model

    @property
    def score(self):
        if not self._score:
            self._score = self.model.evaluate(
                self.dataset.X_test,
                self.dataset.Y_test,
                verbose=int(self.verbose_value)
            )
        return self._score


def main():

    keras1 = KerasNetwork()

    print(keras1.score[0])

    # neural_network = KerasNetwork()
    # network and training

    # data: shuffled and split between train and test sets

    # logger.info('%s train samples \n%s test samples', X_train.shape[0], X_test.shape[0])

    # convert class vectors to binary class matrices

    # 10 outputs
    # final stage is softmax

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

    # score = model.evaluate(X_test, Y_test, verbose=CONFIG.get('VERBOSE'))
    # logger.info("\nTest score: %s \nTest accuracy: %s", score[0], score[1])


if __name__ == "__main__":
    logger = get_logger(__name__)

    with open('chapter1_config.json') as json_data:
        CONFIG = json.load(json_data)['V1']

    main()
