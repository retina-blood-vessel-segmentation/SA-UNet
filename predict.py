import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

try:
    from mlrun import get_or_create_ctx
except ImportError:
    pass
from keras.callbacks import ModelCheckpoint
from tensorflow import ConfigProto, Session

config = ConfigProto()
config.gpu_options.allow_growth = True
sess = Session(config=config)

from util import *
from SA_UNet import SA_UNet
from config import Config


def test(context, model_path, test_images_dir, test_labels_dir, test_masks_dir, output_dir, dataset, threshold, use_fov,
         desired_size, start_neurons, lr, keep_prob, block_size, test_image_height, test_image_width):

    print(f"> Predicting on {dataset} dataset.")

    n_test_images = len(test_labels_dir)
    x_test, y_test = load_files(test_images_dir, test_labels_dir, desired_size, get_label_pattern_for_dataset(dataset),
                                mode='test')
    model = SA_UNet(
        input_size=(desired_size, desired_size, 3),
        start_neurons=start_neurons,
        lr=lr,
        keep_prob=keep_prob,
        block_size=block_size
    )

    if os.path.isfile(model_path):
        model.load_weights(model_path)
        print(f"> Model {model_path} loaded for prediction.")

    model_checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=False)

    y_pred = model.predict(x_test)
    y_pred = crop_to_shape(y_pred, (n_test_images, test_image_height, test_image_width, 1))

    # save predictions
    probability_maps_dir = os.path.join(output_dir, "probability_maps")
    segmentation_masks_dir = os.path.join(output_dir, "segmentation_masks")
    if not os.path.exists(probability_maps_dir):
        os.makedirs(probability_maps_dir)
    if not os.path.exists(segmentation_masks_dir):
        os.makedirs(segmentation_masks_dir)
    print(f'> Probability maps will be saved at {os.path.abspath(probability_maps_dir)}')
    print(f'> Segmentation masks will be saved at {os.path.abspath(segmentation_masks_dir)}')

    for i, y in enumerate(y_pred):
        _, temp = cv2.threshold(y, threshold, 1, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(probability_maps_dir, f"{i}.png"), y * 255)
        cv2.imwrite(os.path.join(segmentation_masks_dir, f"{i}-threshold-{threshold}.png"), temp * 255)
    y_test = list(np.ravel(y_test))

    # load masks if you want statistics to be calculated just for inside fov
    all_masks_data = None
    if use_fov:
        all_masks_data = np.ravel(load_mask_files(test_masks_dir, test_images_dir, get_mask_pattern_for_dataset(dataset)))

    evaluate(
        y_test=y_test,
        y_pred=np.ravel(y_pred),
        threshold=threshold,
        mask_data=all_masks_data,
        use_fov=use_fov,
        result_dir=output_dir
    )


def get_argument(argument, context, args):
    if context is None:
        return args[argument]
    value = context.get_param(argument)
    if value is None:
        return args[argument]
    return value


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # general arguments
    parser.add_argument('--model_path', type=str, help='Path to the h5 file with weights used for prediction.')
    parser.add_argument('--test_images_dir', type=str, help='Relative or absolute path to the images containing test'
                                                            'directories. The directory should contain only test images'
                                                            'since all files are loaded for prediction purposes.')
    # parser.add_argument('--n_test_images', type=int, help='Number of test images.')
    parser.add_argument('--test_labels_dir', type=str,
                        help='Relative or absolute path to the label images directory. The '
                             'directory should contain only test images since all files are '
                             'loaded for prediction purposes.')
    parser.add_argument('--test_masks_dir', type=str,
                        help='Relative or absolute path to the masks images directory. The '
                             'directory should contain only test images since all files are '
                             'loaded for prediction purposes.')
    parser.add_argument('-o', '--output_dir', type=str, help='A directory where prediction results will be saved.')
    parser.add_argument('-d', '--dataset', choices=['DRIVE', 'CHASE', 'DROPS', 'STARE'],
                        help='Can be DRIVE, CHASE or DROPS.')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Lower bound for predicted pixel being '
                                                                           'classified as a blood vessel.')
    parser.add_argument('--use_fov', action='store_const', const=True, default=True)

    args = vars(parser.parse_args())  # access arguments using dictionary syntax

    try:
        context = get_or_create_ctx('train')
    except NameError:
        context = None
        pass

    dataset = get_argument('dataset', context, args)
    test(
        context=context,
        model_path=get_argument('model_path', context, args),
        test_images_dir=get_argument('test_images_dir', context, args),
        test_labels_dir=get_argument('test_labels_dir', context, args),
        test_masks_dir=get_argument('test_masks_dir', context, args),
        output_dir=get_argument('output_dir', context, args),
        test_image_height=Config.datasets[dataset][0],
        test_image_width=Config.datasets[dataset][1],
        dataset=dataset,
        threshold=get_argument('threshold', context, args),
        use_fov=get_argument('use_fov', context, args),
        desired_size=Config.datasets[dataset][2],
        start_neurons=Config.Network.start_neurons,
        lr=Config.Network.learning_rate,
        keep_prob=Config.Network.keep_prob,
        block_size=Config.Network.block_size,
    )