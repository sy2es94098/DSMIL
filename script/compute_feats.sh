#!/bin/bash
DATASET='C16_patch'
WEIGHT='model-v2'
GPU_ID='2 3'
LAYER='instance'
BACKBONE='resnet18'
PRETRAIN_MODEL='simclr'

python compute_feats.py --dataset=${DATASET} --weights=${WEIGHT} \
	--num_classes 1 --gpu_index ${GPU_ID} \
	--norm_layer ${LAYER} --backbone ${BACKBONE} \
	--pretrain_model ${PRETRAIN_MODEL}
