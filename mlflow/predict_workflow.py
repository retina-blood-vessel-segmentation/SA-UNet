import glob
import mlflow

from pathlib import Path

from config import DatasetConfiguration

project_path = Path('..').resolve()
models_root_dir = project_path / 'models'
results_root_dir = project_path / 'results'
datasets_root_dir = project_path / 'data'

python_interpreter = '/home/gorana/miniconda3/envs/saunetpy36/bin/python'
mlflow.set_tracking_uri(str(Path('../../mlflow_db').resolve()))


def predict_all_models():

    trained_models = glob.glob(f"{models_root_dir}/*/saunet.h5")
    datasets_configs = DatasetConfiguration.get_datasets_configuration(datasets_root_dir)

    for model_path in trained_models:
        for dcfg in datasets_configs:

            parameters = {
                'model_path': model_path,
                'test_images_dir': str(project_path / dcfg.test_images_path),
                'test_labels_dir': str(project_path / dcfg.test_labels_path),
                'test_masks_dir': str(project_path / dcfg.test_masks_path),
                'dataset': dcfg.dataset_name,
                'output_dir': str(project_path / results_root_dir / dcfg.dataset_name /
                                  Path(model_path).parent.stem),
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


if __name__ == '__main__':
    predict_all_models()
