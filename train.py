import argparse
import os
import json
import matplotlib.pyplot as plt

from datetime import datetime
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


parser = argparse.ArgumentParser()

# general arguments
parser.add_argument('--model_path', type=str, help='Path to the h5 file where the trained model will be saved.')
parser.add_argument('--weights_path', type=str, default=None, help="Path to the h5 file with network weights.")
parser.add_argument('--train_images_dir', type=str, help='Relative or absolute path to a directory containing training '
                                                         'images. The directory should contain only training images '
                                                         'since all files loaded during the training.')
parser.add_argument('--train_labels_dir', type=str, help='Relative or absolute path to the label images directory. The '
                                                         'directory should contain only training images since all '
                                                         'files are loaded for training purposes.')
parser.add_argument('--val_images_dir', type=str, help="Relative or absolute path to a validation images directory.")
parser.add_argument('--val_labels_dir', type=str, help="Relative or absolute path to a validation labels directory.")
parser.add_argument('-d', '--dataset', choices=['DRIVE', 'CHASE', 'DROPS', 'STARE'], help='Can be DRIVE, CHASE or DROPS.')
parser.add_argument('--dry_run', action='store_const', const=True, default=False)

args = parser.parse_args()

# =================================================================== #
# General arguments
# =================================================================== #
training_images_loc = args.train_images_dir
training_label_loc = args.train_labels_dir
validate_images_loc = args.val_images_dir
validate_label_loc = args.val_labels_dir
transfer_learning_model = args.weights_path
model_path = args.model_path
dataset = args.dataset
dry_run = args.dry_run
# =================================================================== #
# Network parameters
# =================================================================== #
desired_size = Config.datasets[dataset][2]
start_neurons = Config.Network.start_neurons
lr = Config.Network.learning_rate
keep_prob = Config.Network.keep_prob
block_size = Config.Network.block_size
epochs = Config.Network.epochs
batch_size = Config.Network.batch_size
# =================================================================== #


x_train, y_train = load_files(
    training_images_loc,
    training_label_loc,
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
        plt.plot(history.history['acc'])
    except KeyError as e:
        plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('SA-UNet Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='lower right')
    plt.show()
    plt.savefig(Path(model_path).parent / (Path(model_path).stem))
