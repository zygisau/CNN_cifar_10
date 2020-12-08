import datetime

import tensorflow as tf
import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix as sklearn_cm

from math_utils import MathUtils
from plot_utils import PlotUtils
from labels import LABELS


class Model:
    def __init__(self, l_rate, momentum, optimizer, loss):
        self.l_rate = l_rate
        self.momentum = momentum
        self.optimizer = optimizer
        self.loss = loss
        self.history = None
        self.callbacks = []
        self.checkpoint_path = "training_1/cp.ckpt"
        self.model = Sequential()
        self.__define_model()
        self.__define_optimizers()
        self.__initialize_board()
        self.__initialize_model_saver()
        self.model.summary()

    def __define_optimizers(self):
        self.opt = self.optimizer(lr=self.l_rate, momentum=self.momentum)
        self.model.compile(optimizer=self.opt, loss=self.loss, metrics=['accuracy'])

    def __define_model(self):
        self.model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                              input_shape=(32, 32, 3)))
        self.model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(10, activation='softmax'))

    def fit(self, train_data, train_labels, validation_data, n_epoch=100, batch_size=64, verbose=0):
        self.history = self.model.fit(x=train_data, y=train_labels, batch_size=batch_size, epochs=n_epoch,
                                      verbose=verbose, validation_data=validation_data,
                                      callbacks=self.callbacks)

    def evaluate(self, test_data, test_labels, batch_size=64, verbose=0):
        predictions_raw = self.model.predict(test_data, batch_size=batch_size, verbose=verbose)
        predictions_labels_indexes = np.argmax(predictions_raw, axis=1)
        test_labels_indexes = np.argmax(test_labels, axis=1)

        confusion_matrix = sklearn_cm(test_labels_indexes, predictions_labels_indexes)
        PlotUtils.plot_confusion_matrix(confusion_matrix, class_names=LABELS)

        return MathUtils.calculate_loss(test_labels, predictions_raw), MathUtils.calculate_accuracy(test_labels,
                                                                                                    predictions_raw)

    def __initialize_board(self):
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1))

    def __initialize_model_saver(self):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)
        self.callbacks.append(cp_callback)

    def load_model(self):
        self.model.load_weights(self.checkpoint_path)
