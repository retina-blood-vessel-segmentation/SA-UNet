import os

from mlrun import mlconf
from mlrun import new_task, run_local

SOURCE_ROOT_DIR = os.path.abspath('../')

mlconf.dbpath = os.path.abspath('../mlrundb')
mlconf.artifact_path = os.path.abspath('../mlrundb/data')


def predict_on_DRIVE():
    """
    Automate training on DRIVE dataset and SA-Unet with no preloaded weights.
    :return: None
    """
    task = new_task(name='test-drive-data-on-drive-model')\
        .with_param_file(param_file='./config/drive.csv')\
        .with_params(
            model_path="Model/DRIVE/SA_UNet.h5",
            output_dir="results/DRIVE/DRIVE-model",
            use_fov=True
        )\
        .set_label('type', 'test')\
        .set_label('dataset', 'drive')\
        .set_label('fov',  True)

    run_object = run_local(task=task, command='predict.py', workdir=SOURCE_ROOT_DIR)
    run_object.show()
    # run_object.logs()


if __name__ == '__main__':
    predict_on_DRIVE()