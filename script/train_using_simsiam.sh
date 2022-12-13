#!/bin/bash
DATASET=C16_paper_simsiam_24ep
FEATS_SIZE=2048
LR=0.00006
EP=2
GPU_ID='3'
#WEIGHT='weights/simsiam_30ep/1.pth'
python train_tcga.py --dataset=${DATASET} --num_classes 1 --feats_size ${FEATS_SIZE} \
		     --lr ${LR} --num_epochs ${EP} --gpu_index ${GPU_ID} --split 0.1
#		     --aggregator_weights ${WEIGHT}

