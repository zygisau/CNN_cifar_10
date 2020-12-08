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
