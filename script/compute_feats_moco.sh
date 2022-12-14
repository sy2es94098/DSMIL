#!/bin/bash
DATASET='C16_patch'
WEIGHT='contrastive_models/moco/weight/checkpoint_0068.pth.tar'
GPU_ID='2 3'
LAYER='batch'
BACKBONE='resnet50'
PRETRAIN_MODEL='moco'

python compute_feats.py --dataset=${DATASET} --weights=${WEIGHT} \
	--num_classes 1 --gpu_index ${GPU_ID} \
	--norm_layer ${LAYER} --backbone ${BACKBONE} \
	--pretrain_model ${PRETRAIN_MODEL}
