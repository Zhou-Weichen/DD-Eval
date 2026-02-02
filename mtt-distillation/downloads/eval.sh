#!/bin/bash

DATA_PATH='/media/gpu-server-1/4TB_for_data/Zhou/DD/mtt-distillation/dataset/tiny'
LOAD_PATH='/media/gpu-server-1/4TB_for_data/Zhou/DD/mtt-distillation/downloads/tiny_50'
DATASET='Tiny' # CIFAR10 / CIFAR100 / ImageNet / Tiny
SUBSET='imagenette' # only for DATASET='ImageNet'
EVAL_MODE='4'
NUM_EVAL=5
EPOCH_EVAL_TRAIN=1000
LR_NET=0.01
BATCH_TRAIN=256
MODEL='ConvNet'

python eval.py \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --load_path ${LOAD_PATH} \
    --eval_mode ${EVAL_MODE} \
    --lr_net ${LR_NET}  \
    --dsa="True" \
    --dsa_strategy "color_crop_cutout_flip_scale_rotate" \
    # --zca \