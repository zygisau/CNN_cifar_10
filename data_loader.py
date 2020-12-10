import math
import pickle
import numpy as np

from tensorflow.python.keras.utils.np_utils import to_categorical

from math_utils import MathUtils


class Dataset:
    def __init__(self, data_folder):
        num_train_samples = 50000
        self.train_features = np.empty((num_train_samples, 32, 32, 3), dtype='uint8')
        self.train_labels = np.empty((num_train_samples,), dtype='uint8')
        self.validation_features = None
        self.validation_labels = None
        self.test_features = None
        self.test_labels = None
        self.data_folder = data_folder

    def load_data(self, data_parts):
        if math.fsum(data_parts) != 1.0:
            raise ValueError(f'Data parts sum should result in 1. Current result: {math.fsum(data_parts)}')

        self.__load_data(data_parts)
        self.__transform_labels_to_categorical()
        self.__normalize_data()

    def __transform_labels_to_categorical(self):
        self.train_labels = to_categorical(self.train_labels)
        self.test_labels = to_categorical(self.test_labels)
        self.validation_labels = to_categorical(self.validation_labels)

    def __normalize_data(self):
        self.train_features = self.train_features.astype('float32') / 255.0
        self.test_features = self.test_features.astype('float32') / 255.0
        self.validation_features = self.validation_features.astype('float32') / 255.0

    def __load_data(self, data_parts):
        for batch_id in range(1, 6):
            batch = self.__read_file(batch_id)
            self.train_features[(batch_id - 1) * 10000:batch_id * 10000, :, :, :] = batch['data']\
                .reshape((len(batch['data']), 3, 32, 32))\
                .transpose(0, 2, 3, 1)
            self.train_labels[(batch_id - 1) * 10000:batch_id * 10000] = batch['labels']

        split_indices = MathUtils.data_percentage_to_indices(len(self.train_features), data_parts)
        self.train_features, self.test_features, self.validation_features = np.split(self.train_features, split_indices)
        self.train_labels, self.test_labels, self.validation_labels = np.split(self.train_labels, split_indices)
        return (self.train_features, self.train_labels, self.validation_features), (self.test_features, self.test_labels, self.validation_labels)

    def __read_file(self, batch_id):
        with open(self.data_folder + '/data_batch_' + str(batch_id), mode='rb') as file:
            return pickle.load(file, encoding='latin1')

    @staticmethod
    def get_occurrences(dataset):
        unique, unique_idx = np.unique(np.argmax(dataset, axis=1), return_counts=True)
        return dict(zip(unique, unique_idx))

    def get_data_summary(self):
        return f'Training data length: {len(self.train_features)}; Classes distribution:' \
               f' {Dataset.get_occurrences(self.train_labels)}\n' \
               f'Testing data length: {len(self.test_features)}; Classes distribution:' \
               f' {Dataset.get_occurrences(self.test_labels)}\n' \
               f'Validation data length: {len(self.validation_features)}; Classes distribution:' \
               f' {Dataset.get_occurrences(self.validation_labels)}\n' \
