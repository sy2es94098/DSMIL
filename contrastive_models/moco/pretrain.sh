#!/bin/bash

GPU_ID=2,3
BATCH_SIZE=128
DATASET=/data3/ian/dsmil-wsi/WSI/project_tiff
#DATASET=/data3/ian/dsmil-wsi/WSI/C16_paper_tmp_name/single
EP=600
START_EP=23
RESUME=/data3/ian/dsmil-wsi/moco/checkpoint/checkpoint_0022.pth.tar
export CUDA_VISIBLE_DEVICES=${GPU_ID}

python main_moco.py \
  -a resnet50 \
  --lr 0.015 \
  --batch-size ${BATCH_SIZE} \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --mlp --moco-t 0.2 --aug-plus --cos \
  --start-epoch ${START_EP} --epochs ${EP} \
  --resume ${RESUME} \
  ${DATASET}

