#!/bin/bash

# Evaluate STARE dataset on a model trained on CHASE dataset

ROOTDIR=".."
python "${ROOTDIR}/predict.py" \
  --dataset "STARE" \
  --model_path "${ROOTDIR}/Model/CHASE/SA_UNet.h5" \
  --test_images_dir "${ROOTDIR}/STARE/test/images" \
  --n_test_images 8 \
  --test_labels_dir "${ROOTDIR}/STARE/test/labels" \
  --test_masks_dir "${ROOTDIR}/STARE/test/masks" \
  --output_dir "${ROOTDIR}/results/STARE/proba/CHASE-model" \
  --use_fov

