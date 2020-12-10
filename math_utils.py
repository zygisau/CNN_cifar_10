import tensorflow as tf


class MathUtils:
    @staticmethod
    def calculate_loss(test_labels, predictions):
        categorical_cross_entropy = tf.keras.losses.CategoricalCrossentropy()
        return categorical_cross_entropy(test_labels, predictions).numpy()

    @staticmethod
    def calculate_accuracy(test_labels, predictions):
        categorical_accuracy = tf.keras.metrics.CategoricalAccuracy()
        categorical_accuracy.update_state(test_labels, predictions)
        return categorical_accuracy.result().numpy()

    @staticmethod
    def data_percentage_to_indices(data_len, data_parts):
        train_data_index_bound = int(data_len * data_parts[0])
        test_data_index_bound = int(train_data_index_bound + data_len * data_parts[1])
        return [train_data_index_bound, test_data_index_bound]
