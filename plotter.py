import glob
import matplotlib.pyplot as plt
import pickle

from pathlib import Path


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

    RESULTS_PATH = '/home/crvenpaka/ftn/Oftalmologija/segmentacija-mreze/SA-UNet'

    plot_group(RESULTS_PATH)
