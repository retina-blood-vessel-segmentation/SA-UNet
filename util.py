import cv2
import imageio
import math
import numpy as np
import os

from pathlib import Path
from sklearn.metrics import recall_score, roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, jaccard_score


def crop_to_shape(data, shape):
    """
    Crops the array to the given image shape by removing the border (expects a tensor of shape [batches, nx, ny, channels].

    :param data: the array to crop
    :param shape: the target shape
    """
    #

    offset0 = (data.shape[1] - shape[1])//2
    offset1 = (data.shape[2] - shape[2])//2
    if offset0==0:
        if data.shape[1] % 2 == 1 or shape[1] % 2 == 1:
            return data[:, offset0:data.shape[1], offset1:(-offset1)]
        elif data.shape[2] % 2 == 1 or shape[2] % 2 == 1:
            return data[:, offset0:data.shape[1], offset1:(-offset1 - 1)]
        else:
            return data[:, offset0:data.shape[1], offset1:(-offset1)]
    elif offset1==0:
        if data.shape[1] % 2 == 1 or shape[1] % 2 == 1:
            return data[:, offset0:(-offset0 - 1), offset1:data.shape[2]]
        elif data.shape[2] % 2 == 1 or shape[2] % 2 == 1:
            return data[:, offset0:-offset0, offset1:data.shape[2]]
        else:
            return data[:, offset0:-offset0, offset1:data.shape[2]]
    else:
        if data.shape[1] % 2 == 1 or shape[1] % 2 == 1:
            return data[:, offset0:(-offset0 - 1), offset1:(-offset1)]
        elif data.shape[2] % 2 == 1 or shape[2] % 2 == 1:
            return data[:, offset0:-offset0, offset1:(-offset1-1)]
        else:
            return data[:, offset0:-offset0, offset1:(-offset1)]


def evaluate(y_test, y_pred, threshold=0.5, mask_data=None, use_fov=False):
    """
    Calculate numerical metrics based on input predictions image and ground-truth values.

    Calculated metrics are accuracy, precision, sensitivity, specificity, f1 measure, jaccard similarity score, MCC and
    area under ROC curve. Metrics can be calculated for all input predictions or just portions inside field-of-view.
    In later case, use_fov must be set to True and mask data must be provided. All metrics are printed on the standard
    output.

    :param y_test: 1-D array of ground-truth {0, 1} values where 1 represents a pixel of positive class, and 0 a pixel
    of negative class.
    :param y_pred: 1-D array of predictions normalized to [0, 1] interval. Must be of same length as y_test.
    :param threshold: A value in [0, 1] range used to convert y_pred from (0, 1) interval to {0, 1} values.
    :param mask_data: 1-D array of values in (0, 1) range used to mask y_pred values. Must be of same length as y_pred.
    :param use_fov: If True, y_pred will be masked.
    :return: None
    """

    assert len(y_pred) == len(y_test)
    if use_fov:
        assert mask_data is not None
    if mask_data is not None:
        assert len(mask_data) == len(y_test)

    y_pred_threshold = np.ravel(y_pred > threshold)
    if use_fov:
        y_test_inside_fov = [y for m, y in zip(mask_data, y_test) if m > 0.5]
        y_pred_inside_fov = [y for m, y in zip(mask_data, np.ravel(y_pred)) if m > 0.5]
        y_pred_thresh_inside_fov = [y for m, y in zip(mask_data, y_pred_threshold) if m > 0.5]

        assert len(y_test_inside_fov) == len(y_pred_inside_fov) == len(y_pred_thresh_inside_fov)

        y_test = y_test_inside_fov
        y_pred = y_pred_inside_fov
        y_pred_threshold = y_pred_thresh_inside_fov

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()
    N = tn + tp + fn + fp
    S = (tp + fn) / N
    P = (tp + fp) / N

    print('Accuracy:', accuracy_score(y_test, y_pred_threshold))
    print('Sensitivity:', recall_score(y_test, y_pred_threshold))
    print('Specificity', tn / (tn + fp))
    print('Precision: ', precision_score(y_test, y_pred_threshold))
    print('ROCAUC: ', roc_auc_score(y_test, y_pred))
    print("F1: ", 2 * tp / (2 * tp + fn + fp))
    print("Jaccard score: ", jaccard_score(y_test, y_pred_threshold))
    print("MCC:", (tp / N - S * P) / math.sqrt(P * S * (1 - S) * (1 - P)))


def load_files(images_path, label_path, desired_size, label_name_fnc, mode):
    """

    :param path:
    :param desired_size:
    :return:
    """

    images_path = Path(images_path)
    label_path = Path(label_path)

    images = list()
    labels = list()
    for p in images_path.glob('**/*'):
        im = imageio.imread(str(p))
        old_size = im.shape[:2]
        delta_w = desired_size - old_size[1]
        delta_h = desired_size - old_size[0]

        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        images.append(cv2.resize(new_im, (desired_size, desired_size)))

        label = imageio.imread(label_path / label_name_fnc(p), pilmode='L')
        if mode.lower() in ['train', 'validate']:
            new_label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0])
            _, temp = cv2.threshold(new_label, 127, 255, cv2.THRESH_BINARY)
        else:
            _, temp = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)
        labels.append(temp)

    x_data = np.array(images).astype('float32') / 255.
    x_data = np.reshape(x_data, (len(x_data), desired_size, desired_size, 3))
    y_data = np.array(labels).astype('float32') / 255.
    if mode.lower() in ['train', 'validate']:
        y_data = np.reshape(y_data, (len(y_data), desired_size, desired_size, 1))

    return x_data, y_data


def load_mask_files(mask_path, test_path, mask_name_fnc):
    """

    :param mask_path:
    :param test_path:
    :param mask_name_fnc:
    :return:
    """

    test_path = Path(test_path)
    mask_path = Path(mask_path)

    all_masks_data = list()
    for p in test_path.glob("**/*"):
        mask_data = imageio.imread(mask_path / mask_name_fnc(p))
        all_masks_data.append(np.array(mask_data).astype('float32') / 255.)

    return all_masks_data


def get_label_name_drops(image_path):
    return Path(image_path).stem + '.png'


def get_mask_name_drops(image_path):
    return Path(image_path).stem + '.png'


def get_label_name_drive(image_path):
    return Path(image_path).stem.split('_')[0] + '_manual1.png'


def get_mask_name_drive(image_path):
    return Path(image_path).stem + '_mask.gif'


def get_label_name_chase(image_path):
    return Path(image_path).stem + '_1stHO.png'


def get_mask_name_chase(image_path):
    return Path(image_path).stem + '.png'