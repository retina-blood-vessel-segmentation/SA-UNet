#!/bin/bash

# Evaluate CHASE dataset on a model trained on DROPS dataset

ROOTDIR=".."
python "${ROOTDIR}/predict.py" \
  --dataset "CHASE" \
  --model_path "${ROOTDIR}/Model/DROPS/SA_UNet.h5" \
  --test_images_dir "${ROOTDIR}/CHASE/test/images" \
  --n_test_images 8 \
  --test_labels_dir "${ROOTDIR}/CHASE/test/labels" \
  --test_masks_dir "${ROOTDIR}/CHASE/test/masks" \
  --output_dir "${ROOTDIR}/results/CHASE/proba/DROPS-model" \
  --use_fov

