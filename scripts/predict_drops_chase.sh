#!/bin/bash

# Evaluate DROPS dataset on a model trained on CHASE dataset

ROOTDIR=".."
python "${ROOTDIR}/predict.py" \
  --dataset "DROPS" \
  --model_path "${ROOTDIR}/Model/DROPS/SA_UNet_trlrn_CHASE.h5" \
  --test_images_dir "${ROOTDIR}/DROPS/test/images" \
  --n_test_images 4 \
  --test_labels_dir "${ROOTDIR}/DROPS/test/labels" \
  --test_masks_dir "${ROOTDIR}/DROPS/test/masks" \
  --output_dir "${ROOTDIR}/results/DROPS/proba/CHASE-model" \
  --use_fov
  
