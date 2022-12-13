#!/bin/bash

GPU_ID=2,3
BATCH_SIZE=256
#DATASET=/data3/ian/dsmil-wsi/WSI/C16_paper/single
DATASET=/data3/ian/dsmil-wsi/WSI/project_tiff
#DATASET=/data3/ian/dsmil-wsi/WSI/C16_paper_tmp_name/single
EP=600
START_EP=251
RESUME=/data3/ian/dsmil-wsi/simsiam/checkpoint_C16_paper_data/checkpoint_0250.pth.tar
export CUDA_VISIBLE_DEVICES=${GPU_ID}
python main_simsiam.py -a resnet50 --dist-url 'tcp://localhost:10000' --multiprocessing-distributed --world-size 1 --rank 0 \
	      --fix-pred-lr -b ${BATCH_SIZE} --resume ${RESUME}  --start-epoch ${START_EP} --epochs ${EP} ${DATASET}
