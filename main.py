from tensorflow.keras.optimizers import SGD

from model import Model
from data_loader import Dataset

# https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
# https://towardsdatascience.com/cifar-10-image-classification-in-tensorflow-5b501f7dc77c


def train_scenario():
    model.fit(dataset.train_features, dataset.train_labels, n_epoch=n_epoch, batch_size=batch_size, verbose=verbose,
              validation_data=(dataset.validation_features, dataset.validation_labels))


def load_from_file_scenario():
    model.load_model()


if __name__ == '__main__':
    dataset = Dataset(data_folder='./data')
    dataset.load_data(data_parts=[0.7, 0.2, 0.1])

    print(dataset.get_data_summary())

    l_rate = 0.001
    momentum = 0.9
    n_epoch = 100
    batch_size = 64
    verbose = 2
    optimizer = SGD
    loss_func = 'categorical_crossentropy'
    model = Model(l_rate=l_rate, momentum=momentum, optimizer=optimizer, loss=loss_func)

    train_scenario()
    # load_from_file_scenario()

    loss, accuracy = model.evaluate(test_data=dataset.test_features, test_labels=dataset.test_labels,
                                    batch_size=batch_size, verbose=1)
    print(f'Loss: {loss}; Accuracy: {accuracy}')
