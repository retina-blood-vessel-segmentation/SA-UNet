import glob
import matplotlib.pyplot as plt
import pickle
import seaborn

from pathlib import Path


class Plotter:
    """
    Helper methods to plot different plots usefull to track trianing and prediction metrics.
    """

    @staticmethod
    def plot_confusion_matrix(confusion_matrix, title=None):
        """
        Plot confusion matrix.

        :param confusion_matrix: A 2D array matrix produced by scipy confusion_matrix function.
        :param title: A title to a generated plot.
        :return: fig, ax
        """
        fig, ax = plt.subplots(1)
        ax = seaborn.heatmap(confusion_matrix, annot=True, ax=ax,
                        xticklabels=['blood vessel', 'background'],
                        yticklabels=['blood vessel', 'background'])
        if title is None:
            title = 'Confusion matrix'
        ax.set_title(title)

        return fig, ax

    @staticmethod
    def plot_curves(y_data, title, x_label, y_label, x_data=None, legend=None, save=None):
        """
        Plot many curves on the same plot.

        :param x_data: An array list containing y-axis values.
        :param y_data: An array of x values. Must be of same length as arrays from x_data.
        :param y_label: A string label for y-axis.
        :param x_label: A string label for x-axis.
        :param title: The plot title.

        :return: fig, ax
        """
        if x_data is not None:
            assert len(y_data[0]) == len(x_data)
        fig, ax = plt.subplots(1,1)
        for y in y_data:
            if x_data is not None:
                ax.plot(y, x_data)
            else:
                ax.plot(y)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if legend is not None:
            ax.legend(legend, loc='best')
        if save:
            fig.savefig(save)

        return fig, ax

    @staticmethod
    def plot_training_acc(train_acc, validation_acc, save=None):
        fig, ax = Plotter.plot_curves(
            y_data = [train_acc, validation_acc],
            title='Train vs. validation accuracy',
            x_label='Epochs', y_label='Accuracy')
        ax.legend(['trian', 'validate'], loc='best')

        if save:
            fig.savefig(save)

        return fig, ax


def plot_group(path, filter_dataset='*', filter_model='*'):
    """
    Find all result directories containing ROC data for plotting.

    :param path: Absolute or relative path to a root result directory.
    :param filter_dataset: Filter results for some of the datasets.
    :param filter_model: Filter results per model.
    :return: None
    """

    rocs = glob.glob(f'{path}/results/{filter_dataset}/{filter_model}/roc.pickle')

    if len(rocs) == 0:
        return

    plt.figure()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('...')
    plt.legend(loc='lower right')

    for roc in rocs:
        roc = Path(roc)
        dataset = roc.parent.parent
        model = roc.parent

        if filter_model != '*' and filter_dataset != '*':
            label = str(dataset) + 'on ' + str(model)
        elif filter_dataset != '*':
            label = str(model)
        elif filter_model != '*':
            label = str(dataset)
        else:
            label = str(dataset) + 'on ' + str(model)

        with open(roc, 'rb') as roc_file:
            fpr, tpr, _ = pickle.load(roc_file)
            plt.plot(fpr, tpr, lw=2, label=label)
            print(f'Dodata kriva za {label}.')

    plt.show()


if __name__ == '__main__':

    # test roc curve plot
    # RESULTS_PATH = '/home/crvenpaka/ftn/Oftalmologija/segmentacija-mreze/SA-UNet'
    # plot_group(RESULTS_PATH)

    # test Plotter.plot_curves
    from numpy import arange
    y_data = [arange(0, 1, 0.1), arange(0, 1, 0.1) + 1, arange(0, 1, 0.1) + 0.5]
    x_data = arange(0, 1, 0.1)
    fig, ax = Plotter.plot_curves(
        x_data=x_data,
        y_data=y_data,
        title='Experimental title',
        x_label='x label',
        y_label='y label'
    )
    plt.show()

    # test Plotter.plot_training_acc
    fig, ax = Plotter.plot_training_acc(y_data[0], y_data[1], save='./example.png')
    plt.show()

    # test Plotter.plot_confusion_matrix
    confusion_matrix = [[10, 1], [5, 6]]
    fig, ax = Plotter.plot_confusion_matrix(confusion_matrix)
    plt.show()