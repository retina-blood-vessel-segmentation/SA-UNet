import click
import mlflow
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.callbacks import ModelCheckpoint
from tensorflow import ConfigProto, Session

config = ConfigProto()
config.gpu_options.allow_growth = True
sess = Session(config=config)

from util import *
from SA_UNet import SA_UNet


@click.command()
@click.option('--model_path', help='Path to the h5 file with weights used for prediction.')
@click.option('--test_images_dir', help='A directory containing images for prediction.')
@click.option('--test_labels_dir', help='A directory containing corresponding groundtruth images for prediction images.')
@click.option('--test_masks_dir', help='A directory containing corresponding masks for prediction images.')
@click.option('--dataset', type=click.Choice(['DRIVE', 'STARE', 'CHASE', 'DROPS','DRIVE-eval','STARE-eval','CHASE-eval']),
              help='A case-sensitive dataset name that will be used for inference. ')
@click.option('--output_dir', help='Path to the directory where inference results will be saved.')
@click.option('--threshold', type=click.FLOAT, default=0.5, help='Lower bound for predicted pixel being classified as a blood vessel.')
@click.option('--use_fov', is_flag=True, default=True, help='Calculate metrics just for pixels inside FoV.')
@click.option('--start_neurons', type=click.INT, default=16, help='')
@click.option('--lr', type=click.FLOAT, default=1e-3, help='')
@click.option('--keep_prob', type=click.FLOAT, default=1, help='')
@click.option('--block_size',  type=click.INT, default=1, help='')
def test(model_path, test_images_dir, test_labels_dir, test_masks_dir, output_dir, dataset, threshold, use_fov,
         start_neurons, lr, keep_prob, block_size):

    with mlflow.start_run() as run:
        mlflow.log_params({
            'dataset': dataset,
            'model_path': model_path,
            'threshold': threshold,
            'use_fov': use_fov,
            'start_neurons': start_neurons,
            'lr': lr,
            'keep_probability': keep_prob,
            'block_size': block_size
        })

        print(f"> Predicting on {dataset} dataset.")

        desired_size = get_desired_size(dataset)
        n_test_images = len(test_labels_dir)

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

        probability_maps_dir = os.path.join(output_dir, "probability_maps")
        segmentation_masks_dir = os.path.join(output_dir, "segmentation_masks")
        if not os.path.exists(probability_maps_dir):
            os.makedirs(probability_maps_dir)
        if not os.path.exists(segmentation_masks_dir):
            os.makedirs(segmentation_masks_dir)
        print(f'> Probability maps will be saved at {os.path.abspath(probability_maps_dir)}')
        print(f'> Segmentation masks will be saved at {os.path.abspath(segmentation_masks_dir)}')


        flag = True
        bucket = 25
        startat = 0
        while flag:
            x_test, y_test,n = load_files(test_images_dir, test_labels_dir, desired_size, get_label_pattern_for_dataset(dataset),
                                        mode='test',startat=startat,bucket=bucket)
            assert(len(x_test) != 0)

            test_image_width, test_image_height = y_test[0].shape

            model_checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=False)

            y_pred = model.predict(x_test)
            y_pred = crop_to_shape(y_pred, (n_test_images, test_image_height, test_image_width, 1))

            # save predictions

            # mlflow.log_artifact(probability_maps_dir)
            # mlflow.log_artifact(segmentation_masks_dir)

            for i, y in enumerate(y_pred):
                _, temp = cv2.threshold(y, threshold, 1, cv2.THRESH_BINARY)
                cv2.imwrite(os.path.join(probability_maps_dir, f"{i+startat}.png"), y * 255)
                cv2.imwrite(os.path.join(segmentation_masks_dir, f"{i+startat}-threshold-{threshold}.png"), temp * 255)
            y_test = list(np.ravel(y_test))

            # load masks if you want statistics to be calculated just for inside fov
            all_masks_data = None
            if use_fov:
                all_masks_data = np.ravel(load_mask_files(test_masks_dir, test_images_dir, get_mask_pattern_for_dataset(dataset), startat, bucket))

#            odr = os.path.join(output_dir, f"r-{startat}-{n}")
            #if not os.path.exists(odr):
            #    os.makedirs(odr)

#            evaluate(
#                y_test=y_test,
#                y_pred=np.ravel(y_pred),
#                threshold=threshold,
#                mask_data=all_masks_data,
#                use_fov=use_fov,
#                result_dir=output_dir
#            )
            startat = startat + bucket



if __name__ == '__main__':
    test()