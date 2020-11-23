import mlflow
import os
import sys

from config import datasets
from pathlib import Path

project_path = Path('..').resolve()
models_root_dir = project_path / 'models'
results_root_dir = project_path / 'results'
datasets_root_dir = project_path / 'data'
python_interpreter = '/home/gorana/miniconda3/envs/saunetpy36/bin/python'


def train_all_without_transfer_learning(models_root_dir):
    """
    Train models without explicit network weights initialisation
    for all configured datasets.

    :param models_root_dir: A root directory where model subdirectories will be created.
    :return: None
    """
    for dataset_config in datasets:

        parameters = {
            'train_images_dir': str(project_path / dataset_config.train_images_path),
            'train_labels_dir': str(project_path / dataset_config.train_labels_path),
            'val_images_dir': str(project_path / dataset_config.val_images_path),
            'val_labels_dir': str(project_path / dataset_config.val_labels_path),
            'dataset': dataset_config.dataset_name,
            'model_path': str(models_root_dir / dataset_config.dataset_name + '-model' / 'saunet.h5')
        }

        try:
            mlflow.projects.run(
                uri=str(project_path),
                entry_point='train',
                parameters=parameters,
                experiment_name='SAUNet',
                use_conda=False
            )
        except mlflow.exceptions.ExecutionException as e:
            print('mlflow run execution failed.')
            print(e)
            pass


def tune_network_parameters(models_root_dir):
    raise NotImplementedError


if __name__ == '__main__':
    train_all_without_transfer_learning(models_root_dir)

