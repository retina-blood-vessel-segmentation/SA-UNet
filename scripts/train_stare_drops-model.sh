#!/bin/bash

# Train the network on STARE dataset using network weights from DROPS dataset

ROOTDIR=".."
DATASET="STARE"
python "${ROOTDIR}/train.py" \
  --dataset ${DATASET} \
  --weights_path "${ROOTDIR}/Model/DROPS/SA_UNet.h5" \
  --model_path "${ROOTDIR}/Model/${DATASET}/SA_UNet_trlrn_DROPS.h5" \
  --train_images_dir "${ROOTDIR}/${DATASET}/train/images" \
  --val_images_dir "${ROOTDIR}/${DATASET}/validate/images" \
  --train_labels_dir "${ROOTDIR}/${DATASET}/train/labels" \
  --val_labels_dir "${ROOTDIR}/${DATASET}/validate/labels" \
  --dry_run

