import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.callbacks import  ModelCheckpoint
from tensorflow import ConfigProto, Session

config = ConfigProto()
config.gpu_options.allow_growth = True
sess = Session(config=config)

from util import *
from SA_UNet import SA_UNet


# TODO modify the section above
# =================================================================== #
model_name = "Model/CHASE/SA_UNet.h5"
data_location = './'
testing_images_loc = os.path.join(data_location, 'CHASE/test/images/')
testing_label_loc = os.path.join(data_location, 'CHASE/test/labels/')
fov_masks_loc = os.path.join(data_location, 'CHASE/test/masks')
output_dir = 'CHASE/test/results'
n_test_images = 8
image_width, image_height = (960, 999)
desired_size = 1008
threshold = 0.5
use_fov = True
# =================================================================== #


def get_label_name(image_path):
    return Path(image_path).stem + '_1stHO.png'


def get_mask_name(image_path):
    return Path(image_path).stem + '.png'


print("> Predicting on CHASE dataset.")

x_test, y_test = load_test_files(testing_images_loc, testing_label_loc, desired_size, get_label_name)

model = SA_UNet(input_size=(desired_size, desired_size, 3), start_neurons=16, lr=1e-4, keep_prob=1, block_size=1)

if os.path.isfile(model_name):
    model.load_weights(model_name)
    print(f"> Model {model_name} loaded for prediction.")

model_checkpoint = ModelCheckpoint(model_name, monitor='val_accuracy', verbose=1, save_best_only=False)

y_pred = model.predict(x_test)
y_pred = crop_to_shape(y_pred,(n_test_images, image_width, image_height, 1))

# save predictions
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i, y in enumerate(y_pred):
    _, temp = cv2.threshold(y, threshold, 1, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(output_dir, f"{i}.png"), y * 255)
    cv2.imwrite(os.path.join(output_dir, f"{i}-thresholded.png"), temp * 255)
y_test = list(np.ravel(y_test))

# load masks if you want statistics to be calculate just for inside fov
all_masks_data = None
if use_fov:
    all_masks_data = np.ravel(load_mask_files(fov_masks_loc, testing_images_loc, get_mask_name))

evaluate(y_test=y_test, y_pred=np.ravel(y_pred), threshold=threshold, mask_data=all_masks_data, use_fov=use_fov)
