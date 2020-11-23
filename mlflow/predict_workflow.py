import mlflow

from config import DatasetConfiguration

project_path = Path('..').resolve()
models_root_dir = project_path / 'models'
results_root_dir = project_path / 'results'
datasets_root_dir = project_path / 'data'
python_interpreter = '/home/gorana/miniconda3/envs/saunetpy36/bin/python'


def predict_all_models(models_root_dir, exclude=None):

    dataset_configurations = DatasetConfiguration.get_datasets_configuration(datasets_root_dir)
    for dataset_config in dataset_configurations:
        if dataset_config.dataset_name == any(exclude):
            continue

        parameters = {
            'model_path': None # todo
        }

        try:
            mlflow.projects.run(
                uri=str(project_path),
                entry_point='test',
                parameters=parameters,
                experiment_name='SAUNet',
                use_conda=False
            )
        except mlflow.exceptions.ExecutionException as e:
            print('mlflow run execution failed.')
            print(e)
            pass

