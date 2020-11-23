import click
import os
import json
import mlflow
import matplotlib.pyplot as plt

from datetime import datetime
from keras.callbacks import TensorBoard, ModelCheckpoint
from pathlib import Path
from tensorflow import ConfigProto, Session

config = ConfigProto()
config.gpu_options.allow_growth = True
sess = Session(config=config)

from SA_UNet import SA_UNet
from util import load_files, get_desired_size
from util import get_label_pattern_for_dataset
from plotter import Plotter


@click.command()
@click.option('--train_images_dir', help='A path to a directory with training images.')
@click.option('--train_labels_dir', help='A path to a directory with groundtruths corresponding to training images.')
@click.option('--val_images_dir', help='A path to a directory with validation images.')
@click.option('--val_labels_dir', help='A path to a directory with groundtruths corresponding to validation images')
@click.option('--dataset', type=click.Choice(['DRIVE', 'STARE', 'CHASE', 'DROPS'], case_sensitive=True),
              help='A dataset on which to train the network.')
@click.option('--model_path', help='Path to the h5 file where the trained model will be saved.')
@click.option('--transfer_learning_model', default=None, help='Path to the h5 file with network weights for initialization.')
@click.option('--start_neurons', type=click.INT, default=16, help='')
@click.option('--epochs', type=click.INT, default=100, help='')
@click.option('--lr', type=click.FLOAT, default=1e-3, help='Training learning rate.')
@click.option('--keep_prob', type=click.FLOAT, default=1, help='')
@click.option('--block_size', type=click.INT, default=1, help='')
@click.option('--batch_size', type=click.INT, default=2, help='')
@click.option('--dry_run', is_flag=True, help='Run training script and skip model training.')
def train(train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, model_path, dataset,
          start_neurons, lr, keep_prob, block_size, epochs, batch_size, dry_run, transfer_learning_model):
    mlflow.tensorflow.autolog(every_n_iter=1)
    with mlflow.start_run() as run:
        mlflow.log_params({
            'dataset': dataset,
            'start_neurons': start_neurons,
            'lr': lr,
            'epochs': epochs,
            'keep_probability': keep_prob,
            'block_size': block_size,
            'batch_size': batch_size
        })
        desired_size = get_desired_size(dataset)
        x_train, y_train = load_files(
            train_images_dir,
            train_labels_dir,
            desired_size,
            get_label_pattern_for_dataset(dataset),
            mode='train'
        )
        x_validate, y_validate = load_files(
            val_images_dir,
            val_labels_dir,
            desired_size,
            get_label_pattern_for_dataset(dataset),
            mode='validate')

        logdir = Path(f"logs/{dataset}") / datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(
            log_dir=logdir,
            histogram_freq=0,
            write_graph=True,
            write_images=True
        )

        model = SA_UNet(
            input_size=(desired_size, desired_size, 3),
            start_neurons=start_neurons,
            lr=lr,
            keep_prob=keep_prob,
            block_size=block_size
        )

        if transfer_learning_model is not None:
            if os.path.isfile(transfer_learning_model):
                print(f"> Initializing the network using weight of model {transfer_learning_model}.")
                model.load_weights(transfer_learning_model)
            else:
                print("> Transfer learning demanded, but no weights are provided.")
                print("> Exiting.")
                exit(1)

        # create directory structure where a trained model will be saved
        if not Path(model_path).parent.exists():
            Path(model_path).parent.mkdir(parents=True)

        model_checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True)

        print(f"> Training the network on dataset {dataset}.")
        if not dry_run:
            history = model.fit(x_train, y_train,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_data=(x_validate, y_validate),
                                shuffle=True,
                                callbacks=[
                                    tensorboard_callback,
                                    model_checkpoint
                                ])
        else:
            print('> Skipping network training (dry run).')

        # save network training parameters alongside model
        train_setup = dict()
        train_setup['desired_size'] = desired_size
        train_setup['learning_rate'] = lr
        train_setup['keep_probability'] = keep_prob
        train_setup['block_size'] = block_size
        train_setup['epochs'] = epochs
        train_setup['batch_size'] = batch_size
        with open(Path(model_path).parent / (Path(model_path).stem + ".setup"), 'w') as fout:
            json.dump(train_setup, fout)

        if not dry_run:
            try:
                train_acc = history.history['acc']
            except KeyError as e:
                train_acc = history.history['accuracy']
            validation_acc = history.history['val_accuracy']
            plt.plot(train_acc)
            plt.plot(validation_acc)
            plt.title('SA-UNet Accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'validate'], loc='lower right')

            save_dir = Path(model_path).parent / (Path(model_path).stem)
            fig, ax = Plotter.plot_training_acc(train_acc, validation_acc, save=str(save_dir) + ".png")
            mlflow.log_artifact(str(save_dir) + '.png')

            # TODO log plot to artefacts


if __name__ == '__main__':
    train()