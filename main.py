from csv_writer import CSVWriter
from image_saver import ImageSaver
from labels import LABELS
from model import Model
from data_loader import Dataset
from options_loader import load_P1_options, load_P2_options, load_P3_options

import numpy as np

# https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
# https://towardsdatascience.com/cifar-10-image-classification-in-tensorflow-5b501f7dc77c


def train_scenario():
    model.fit(dataset.train_features, dataset.train_labels, n_epoch=n_epoch, batch_size=batch_size, verbose=verbose,
              validation_data=(dataset.validation_features, dataset.validation_labels))


def load_from_file_scenario():
    model.load_model()


def print_first_30_photos(preds):
    csv_writer = CSVWriter('first_predictions.csv')
    image_saver = ImageSaver('images/')
    prediction_labels = np.argmax(preds, axis=1)
    test_labels = np.argmax(dataset.test_labels, axis=1)
    test_features = dataset.test_features
    csv_writer.append_to_file(['#', 'Paveikslėlis', 'Nuspėta klasė', 'Tikroji klasė'])
    for index in range(30):
        csv_writer.append_to_file([index + 1, '', LABELS[prediction_labels[index]], LABELS[test_labels[index]]])
        image_saver.plt.imshow(test_features[index])
        image_saver.save_image(index)


if __name__ == '__main__':
    dataset = Dataset(data_folder='./data')
    dataset.load_data(data_parts=[0.7, 0.2, 0.1])

    print(dataset.get_data_summary())

    l_rate, momentum, n_epoch, batch_size, verbose, optimizer, loss_func = load_P1_options()
    model = Model(l_rate=l_rate, momentum=momentum, optimizer=optimizer, loss=loss_func)

    # train_scenario()
    load_from_file_scenario()

    loss, accuracy, predictions = model.evaluate(test_data=dataset.test_features, test_labels=dataset.test_labels,
                                                 batch_size=batch_size, verbose=1)
    print(f'Loss: {loss}; Accuracy: {accuracy}')
    
    print_first_30_photos(predictions)
