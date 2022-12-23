#!/bin/bash

DATASET=C16_paper_simsiam_24ep
FEATS_SIZE=2048
LR=0.0002
EP=5
GPU_ID='2 3'
SEED=9

python train_aggregator.py --dataset=${DATASET} --num_classes 1 --feats_size ${FEATS_SIZE} \
						--lr ${LR} --num_epochs ${EP} --gpu_index ${GPU_ID} --split 0.1 --seed ${i}
