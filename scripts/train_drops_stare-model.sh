#!/bin/bash

# Train the network on DROPS dataset using network weights from STARE dataset

ROOTDIR=".."
DATASET="DROPS"
python "${ROOTDIR}/train.py" \
  --dataset ${DATASET} \
  --weights_path "${ROOTDIR}/Model/STARE/SA_UNet.h5" \
  --model_path "${ROOTDIR}/Model/${DATASET}/SA_UNet_trlrn_STARE.h5" \
  --train_images_dir "${ROOTDIR}/${DATASET}/train/images" \
  --val_images_dir "${ROOTDIR}/${DATASET}/validate/images" \
  --train_labels_dir "${ROOTDIR}/${DATASET}/train/labels" \
  --val_labels_dir "${ROOTDIR}/${DATASET}/validate/labels" \
  --dry_run

