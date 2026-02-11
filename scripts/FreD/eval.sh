#!/bin/bash

DATASET='CIFAR100'        # CIFAR10 / CIFAR100 / ImageNet / Tiny
SUBSET='imagenette'      # only for ImageNet
MODEL='ConvNet'
EVAL_MODE='S'
NUM_EVAL=5
IPC=1

EPOCH_EVAL_TRAIN=1000
BATCH_TRAIN=256
BATCH_REAL=256
LR_TEACHER=0.01

DATA_PATH='./dataset'
LOAD_PATH='./download/cifar100'  
DSA='True'
DSA_STRATEGY='color_crop_cutout_flip_scale_rotate'

SEED=0
# ZCA_FLAG='--zca'

python eval.py \
  --dataset ${DATASET} \
  --subset ${SUBSET} \
  --model ${MODEL} \
  --eval_mode ${EVAL_MODE} \
  --num_eval ${NUM_EVAL} \
  --ipc ${IPC} \
  --epoch_eval_train ${EPOCH_EVAL_TRAIN} \
  --batch_train ${BATCH_TRAIN} \
  --batch_real ${BATCH_REAL} \
  --lr_teacher ${LR_TEACHER} \
  --data_path ${DATA_PATH} \
  --load_path ${LOAD_PATH} \
  --seed ${SEED} \
  --dsa ${DSA} \
  --dsa_strategy ${DSA_STRATEGY} \
  ${ZCA_FLAG}
