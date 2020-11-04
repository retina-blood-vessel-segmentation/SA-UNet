import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.callbacks import ModelCheckpoint
from tensorflow import ConfigProto, Session

config = ConfigProto()
config.gpu_options.allow_growth = True
sess = Session(config=config)

from util import *
from SA_UNet import SA_UNet
from config import Config


parser = argparse.ArgumentParser()

# general arguments
parser.add_argument('--model_path', type=str, help='Path to the h5 file with weights used for prediction.')
parser.add_argument('--test_images_dir', type=str, help='Relative or absolute path to the images containing test'
                                                        'directories. The directory should contain only test images'
                                                        'since all files are loaded for prediction purposes.')
parser.add_argument('--n_test_images', type=int, help='Number of test images.')
parser.add_argument('--test_labels_dir', type=str, help='Relative or absolute path to the label images directory. The '
                                                        'directory should contain only test images since all files are '
                                                        'loaded for prediction purposes.')
parser.add_argument('--test_masks_dir', type=str, help='Relative or absolute path to the masks images directory. The '
                                                       'directory should contain only test images since all files are '
                                                       'loaded for prediction purposes.')
parser.add_argument('-o', '--output_dir', type=str, help='A directory where prediction results will be saved.')
parser.add_argument('-d', '--dataset', choices=['DRIVE', 'CHASE', 'DROPS', 'STARE'], help='Can be DRIVE, CHASE or DROPS.')
parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Lower bound for predicted pixel being '
                                                                       'classified as a blood vessel.')
parser.add_argument('--use_fov', action='store_const', const=True, default=True)

args = parser.parse_args()

# =================================================================== #
# General arguments
# =================================================================== #
dataset = args.dataset
model_name = args.model_path
testing_images_loc = args.test_images_dir
testing_label_loc = args.test_labels_dir
fov_masks_loc = args.test_masks_dir
output_dir = args.output_dir
n_test_images = args.n_test_images
# image_width, image_height = (640, 480)
image_width, image_height = Config.datasets[dataset][:2]
threshold = args.threshold
use_fov = args.use_fov
# =================================================================== #
# Network parameters
# =================================================================== #
desired_size = Config.datasets[dataset][2]
start_neurons = Config.Network.start_neurons
lr = Config.Network.learning_rate
keep_prob = Config.Network.keep_prob
block_size = Config.Network.block_size
# =================================================================== #


print(f"> Predicting on {dataset} dataset.")

x_test, y_test = load_files(testing_images_loc, testing_label_loc, desired_size, get_label_pattern_for_dataset(dataset),
                            mode='test')
model = SA_UNet(
    input_size=(desired_size, desired_size, 3),
    start_neurons=start_neurons,
    lr=lr,
    keep_prob=keep_prob,
    block_size=block_size
)

if os.path.isfile(model_name):
    model.load_weights(model_name)
    print(f"> Model {model_name} loaded for prediction.")

model_checkpoint = ModelCheckpoint(model_name, monitor='val_accuracy', verbose=1, save_best_only=False)

y_pred = model.predict(x_test)
y_pred = crop_to_shape(y_pred, (n_test_images, image_height, image_width, 1))

# save predictions
if not os.path.exists(output_dir):
    probability_maps_dir = os.path.join(output_dir, "probability_maps")
    segmentation_masks_dir = os.path.join(output_dir, "segmentation_masks")
    os.makedirs(probability_maps_dir, exist_ok=True)
    os.makedirs(segmentation_masks_dir, exist_ok=True)

for i, y in enumerate(y_pred):
    _, temp = cv2.threshold(y, threshold, 1, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(output_dir, f"{i}.png"), y * 255)
    cv2.imwrite(os.path.join(output_dir, f"{i}-thresholded.png"), temp * 255)
y_test = list(np.ravel(y_test))

# load masks if you want statistics to be calculated just for inside fov
all_masks_data = None
if use_fov:
    all_masks_data = np.ravel(load_mask_files(fov_masks_loc, testing_images_loc, get_mask_pattern_for_dataset(dataset)))

evaluate(
    y_test=y_test,
    y_pred=np.ravel(y_pred),
    threshold=threshold,
    mask_data=all_masks_data,
    use_fov=use_fov,
    result_dir=output_dir
)
