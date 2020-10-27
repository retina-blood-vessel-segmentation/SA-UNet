#!/bin/bash

# Evaluate DRIVE dataset on a model trained on CHASE dataset

ROOTDIR=".."
python "${ROOTDIR}/predict.py" \
  --dataset "DRIVE" \
  --model_path "${ROOTDIR}/Model/CHASE/SA_UNet.h5" \
  --test_images_dir "${ROOTDIR}/DRIVE/test/images" \
  --n_test_images 20 \
  --test_labels_dir "${ROOTDIR}/DRIVE/test/labels" \
  --test_masks_dir "${ROOTDIR}/DRIVE/test/masks" \
  --output_dir "${ROOTDIR}/results/DRIVE/proba/CHASE-model" \
  --use_fov

