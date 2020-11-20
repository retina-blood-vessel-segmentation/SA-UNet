import argparse
import os
import json
import matplotlib.pyplot as plt

from datetime import datetime
try:
    from mlrun import get_or_create_ctx
except ImportError:
    pass
from keras.callbacks import TensorBoard, ModelCheckpoint
from pathlib import Path
from tensorflow import ConfigProto, Session

config = ConfigProto()
config.gpu_options.allow_growth = True
sess = Session(config=config)

from SA_UNet import SA_UNet
from util import load_files
from util import get_label_pattern_for_dataset
from config import Config


def train(context, training_images_loc, training_labels_loc, validate_images_loc, validate_label_loc, model_path,
          dataset, desired_size, start_neurons, lr, keep_prob, block_size, epochs, batch_size, dry_run=False,
          transfer_learning_model=None):

    x_train, y_train = load_files(
        training_images_loc,
        training_labels_loc,
        desired_size,
        get_label_pattern_for_dataset(dataset),
        mode='train'
    )
    x_validate, y_validate = load_files(
        validate_images_loc,
        validate_label_loc,
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

    # save training accuracy plot alongside model
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
        plt.savefig(Path(model_path).parent / (Path(model_path).stem))


def get_argument(argument, context, args, config):
    if context is None:
        if args.get(argument) is not None:
            return args.get(argument)
        elif config.get(argument) is not None:
            return config.get(argument)
        else:
            return None
    value = context.get_param(argument)
    if value is None:
        if args.get(argument) is not None:
            return args.get(argument)
        elif config.get(argument) is not None:
            return config.get(argument)
        else:
            return None
    return value


if __name__ == '__main__':

    # parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, help='Path to the h5 file where the trained model will be saved.')
    parser.add_argument('--weights_path', type=str, default=None, help="Path to the h5 file with network weights.")
    parser.add_argument('--train_images_dir', type=str,
                        help='Relative or absolute path to a directory containing training '
                             'images. The directory should contain only training images '
                             'since all files loaded during the training.')
    parser.add_argument('--train_labels_dir', type=str,
                        help='Relative or absolute path to the label images directory. The '
                             'directory should contain only training images since all '
                             'files are loaded for training purposes.')
    parser.add_argument('--val_images_dir', type=str,
                        help="Relative or absolute path to a validation images directory.")
    parser.add_argument('--val_labels_dir', type=str,
                        help="Relative or absolute path to a validation labels directory.")
    parser.add_argument('-d', '--dataset', choices=['DRIVE', 'CHASE', 'DROPS', 'STARE'],
                        help='Can be DRIVE, CHASE or DROPS.')
    parser.add_argument('--dry_run', action='store_const', const=True, default=False)

    args = vars(parser.parse_args())  # access arguments using dictionary syntax

    # parse configuration file
    config = json.load(open("config.json"))

    # get mlrun context
    try:
        context = get_or_create_ctx('train')
    except NameError:
        context = None
        pass

    dataset = get_argument('dataset', context, args, config)
    # quick patch for desired_size which should be provided alongside dataset used for training...
    # maybe not the best way to do this, but it will have to do for now
    if dataset == 'DRIVE':
        config['desired_size'] = 592
    else:
        config['desired_size'] = 1008

train(
        context=context,
        training_images_loc=get_argument('train_images_dir', context, args, config),
        training_labels_loc=get_argument('train_labels_dir', context, args, config),
        validate_images_loc=get_argument('val_images_dir', context, args, config),
        validate_label_loc=get_argument('val_labels_dir', context, args, config),
        transfer_learning_model=get_argument('weights_path', context, args, config),
        model_path=get_argument('model_path', context, args, config),
        dataset=dataset,
        dry_run=get_argument('dry_run', context, args, config),
        desired_size=get_argument('desired_size', context, args, config),
        start_neurons=get_argument('start_neurons', context, args, config),
        lr=get_argument('learning_rate', context, args, config),
        keep_prob=get_argument('keep_probability', context, args, config),
        block_size=get_argument('block_size', context, args, config),
        epochs=get_argument('epochs', context, args, config),
        batch_size=get_argument('batch_size', context, args, config)
    )
