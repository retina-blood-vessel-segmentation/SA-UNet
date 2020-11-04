import os
import matplotlib.pyplot as plt

from datetime import datetime
from keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow import ConfigProto, Session
config = ConfigProto()
config.gpu_options.allow_growth = True
sess = Session(config=config)

from SA_UNet import SA_UNet
from util import load_files
from util import get_label_name_drive


# TODO location and model parameters
data_location = ''
training_images_loc = os.path.join(data_location, '../DRIVE/train/images/')
training_label_loc = os.path.join(data_location, '../DRIVE/train/labels/')
validate_images_loc = os.path.join(data_location, '../DRIVE/validate/images/')
validate_label_loc = os.path.join(data_location, '../DRIVE/validate/labels/')
transfer_learning_model = None
model_path = "Model/DRIVE/SA_UNet.py"

# TODO network training parameters
lr = 1e-3
start_neurons = 16
keep_prob = 0.82
block_size = 7
desired_size = 592
epochs = 100
batch_size = 2


x_train, y_train = load_files(training_images_loc, training_label_loc, desired_size, get_label_name_drive,
                              mode='train')
x_validate, y_validate = load_files(validate_images_loc, validate_label_loc, desired_size, get_label_name_drive,
                                    mode='validate')

logdir = "logs/DRIVE/" + datetime.now().strftime("%Y%m%d-%H%M%S")
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
model.summary()

if transfer_learning_model is not None:
    if os.path.isfile(transfer_learning_model):
        print(f"> Initializing the network using weight of model {transfer_learning_model}.")
        model.load_weights(transfer_learning_model)
    else:
        print("Transfer learning demanded, but no weights are provided.")
        print("Exiting.")
        exit(1)

model_checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=False)


history = model.fit(x_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_validate, y_validate),
                shuffle=True,
                callbacks=[
                    tensorboard_callback,
                    model_checkpoint
                ])

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_accuracy'])
plt.title('SA-UNet Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='lower right')
plt.show()


