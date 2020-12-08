import itertools
import sys
import matplotlib.pyplot as plt
import numpy as np


class PlotUtils:
    @staticmethod
    def plot_confusion_matrix(confusion_matrix, class_names):
        plt.figure(figsize=(8, 8))
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        labels = np.around(confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis], decimals=2)

        threshold = confusion_matrix.max() / 2.
        for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
            color = "white" if confusion_matrix[i, j] > threshold else "black"
            plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label', position=(.05, 0.5))
        plt.xlabel('Predicted label')
        filename = sys.argv[0].split('/')[-1]
        plt.savefig(filename + '_plot.png')
        plt.close()
