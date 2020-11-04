#!/bin/bash

# Train the network on DRIVE dataset

ROOTDIR=".."
DATASET="DRIVE"
python "${ROOTDIR}/train.py" \
  --dataset ${DATASET} \
  --model_path "${ROOTDIR}/Model/${DATASET}/SA_UNet_test.h5" \
  --train_images_dir "${ROOTDIR}/${DATASET}/train/images" \
  --val_images_dir "${ROOTDIR}/${DATASET}/validate/images" \
  --train_labels_dir "${ROOTDIR}/${DATASET}/train/labels" \
  --val_labels_dir "${ROOTDIR}/${DATASET}/validate/labels" \
  --dry_run

