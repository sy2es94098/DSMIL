#!/bin/bash
DATASET=C16_paper_simsiam_24ep
FEATS_SIZE=2048
LR=0.0002
EP=200
GPU_ID='0 1'
#WEIGHT='weights/simsiam_30ep/1.pth'
python test_aggregator.py --dataset=${DATASET} --num_classes 1 --feats_size ${FEATS_SIZE} \
		     --lr ${LR} --num_epochs ${EP} --gpu_index ${GPU_ID} --split 0.2
#		     --aggregator_weights ${WEIGHT}

