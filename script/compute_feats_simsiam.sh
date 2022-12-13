#!/bin/bash
#DATASET='project_tiff'
#WEIGHT='simsiam/checkpoint_0410.pth.tar'
DATASET='C16_testing'
WEIGHT='simsiam/checkpoint_C16_paper_data/checkpoint_0073.pth.tar'
GPU_ID='0 1'
LAYER='batch'
BACKBONE='resnet50'
python compute_feats_simsiam.py --dataset=${DATASET} --weights=${WEIGHT} \
	--num_classes 1 --gpu_index ${GPU_ID} \
	--norm_layer ${LAYER} --backbone ${BACKBONE}
