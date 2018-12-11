from keras.optimizers import SGD, Adam


class KerasBaseModel:
    nb_epoch = 200
    batch_size = 128
    verbose_value = 1
    nb_classes = 10
    n_hidden = 128
    validation_split = 0.2
    reshaped = 784
    _optimizer = SGD()

    def __init__(self):
        self._optimizer = None

    @property
    def model_epoch(self):
        return self.nb_epoch

    @model_epoch.setter
    def model_epoch(self, epoch_input):
        self.nb_epoch = epoch_input

    @property
    def verbose(self):
        return self.verbose_value

    @verbose.setter
    def verbose(self, verbose_input):
        verbose_allowed_values = [1, 2, 3]
        if verbose_input in verbose_allowed_values:
            self.verbose_value = verbose_input

    @property
    def model_description(self):

        description = "Keras chapter 1 model " \
                      "\n Epoch: {0}" \
                      "\n Batch size: {1}" \
                      "\n Verbose {2}" \
                      "\n Classes {3}" \
                      "\n Hidden {4}" \
                      "\n Validation split {5}" \
                      "\n Reshaped {6}".format(
            self.nb_epoch,
            self.batch_size,
            self.verbose_value,
            self.nb_classes,
            self.n_hidden,
            self.validation_split,
            self.reshaped)

        return description

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, input_optimizer):
        # TODO add Adam to import
        allowed_optimizers = {'SGD': SGD, 'Adam': Adam}
        self._optimizer = allowed_optimizers.get(input_optimizer, SGD())
