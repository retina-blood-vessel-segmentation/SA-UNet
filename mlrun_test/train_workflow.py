import os

from mlrun import mlconf
from mlrun import new_task, run_local


SOURCE_ROOT_DIR = '../'

mlconf.dbpath = os.path.abspath('../mlrundb')
mlconf.artifact_path = os.path.abspath('../mlrundb/data')


def train_on_DRIVE():
    """
    Automate training on DRIVE dataset and SA-Unet with no preloaded weights.
    :return: None
    """
    task = new_task(name='train-drive-no-weights')\
        .with_param_file(param_file='./config/drive.csv')\
        .with_params(
            model_path="Model/DRIVE/SA_UNet.h5"
        )\
        .set_label('type', 'train')\
        .set_label('dataset', 'drive')\
        .set_label('preloaded_weights', 'no')

    run_object = run_local(task=task, command='train.py', workdir=SOURCE_ROOT_DIR)
    run_object.show()


if __name__ == '__main__':
    train_on_DRIVE()