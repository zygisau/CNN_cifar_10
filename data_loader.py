import pickle
import numpy as np

from tensorflow.python.keras.utils.np_utils import to_categorical


class Dataset:
    def __init__(self, data_folder):
        num_train_samples = 50000
        self.train_features = np.empty((num_train_samples, 32, 32, 3), dtype='uint8')
        self.train_labels = np.empty((num_train_samples,), dtype='uint8')
        self.test_features = None
        self.test_labels = None
        self.data_folder = data_folder

    def load_data(self, train_data_part):
        self.__load_data(train_data_part)
        self.__transform_labels_to_categorical()
        self.__normalize_data()

    def __transform_labels_to_categorical(self):
        self.train_labels = to_categorical(self.train_labels)
        self.test_labels = to_categorical(self.test_labels)

    def __normalize_data(self):
        self.train_features = self.train_features.astype('float32') / 255.0
        self.test_features = self.test_features.astype('float32') / 255.0

    def __load_data(self, train_data_part):
        for batch_id in range(1, 6):
            batch = self.__read_file(batch_id)
            self.train_features[(batch_id - 1) * 10000:batch_id * 10000, :, :, :] = batch['data']\
                .reshape((len(batch['data']), 3, 32, 32))\
                .transpose(0, 2, 3, 1)
            self.train_labels[(batch_id - 1) * 10000:batch_id * 10000] = batch['labels']

        self.train_features, self.test_features = np.split(self.train_features,
                                                           [int(len(self.train_features)*train_data_part)])
        self.train_labels, self.test_labels = np.split(self.train_labels,
                                                       [int(len(self.train_labels) * train_data_part)])
        return (self.train_features, self.train_labels), (self.test_features, self.test_labels)

    def __read_file(self, batch_id):
        with open(self.data_folder + '/data_batch_' + str(batch_id), mode='rb') as file:
            return pickle.load(file, encoding='latin1')
