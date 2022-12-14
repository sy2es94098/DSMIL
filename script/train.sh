#!/bin/bash
DATASET=C16_paper5
FEATS_SIZE=512
LR=0.00006
EP=10
GPU_ID='3'
SEED=9
#WEIGHT='weights/simsiam_30ep/1.pth'
python train_tcga.py --dataset=${DATASET} --num_classes 1 --feats_size ${FEATS_SIZE} \
		     --lr ${LR} --num_epochs ${EP} --gpu_index ${GPU_ID} --split 0.1 --seed ${SEED}
#		     --aggregator_weights ${WEIGHT}

