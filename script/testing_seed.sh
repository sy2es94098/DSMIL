#!/bin/bash
#: << EOF
seed="64 921 911 77 1894 52 36 484 1 2 5 487 55 44 18 99 87"
DATASET=C16_paper_simsiam_24ep
FEATS_SIZE=2048
LR=0.0002
EP=5
GPU_ID='2 3'
#WEIGHT='weights/simsiam_30ep/1.pth'

for i in $seed; do
	python train_tcga.py --dataset=${DATASET} --num_classes 1 --feats_size ${FEATS_SIZE} \
        	             --lr ${LR} --num_epochs ${EP} --gpu_index ${GPU_ID} --split 0.1 --seed ${i}
#               	     --aggregator_weights ${WEIGHT}   
done

#EOF


base=1
BAG_PATH=test-c16/patches
MAP_PATH=test-c16/map_paper_m_select__
EMBEDDER=embedder/C16_paper_simsiam_24ep/embedder.pth
#EMBEDDER=simsiam/checkpoint_0004.pth.tar
AGGREGATOR=weights_C16/10262022/
EXT=jpeg
FEATS_SIZE=2048

for ((i=0;i<17;i++)); do
	AGGR_WEIGHT=${AGGREGATOR}$((${base}+${i}))".pth"
	DIR=${MAP_PATH}$((${base}+${i}))
	echo ${AGGR_WEIGHT}
	python testing_cts_vis_critical.py --num_classes 1 --bag_path ${BAG_PATH} --map_path ${DIR} --embedder_weights ${EMBEDDER} --aggregator_weights ${AGGR_WEIGHT} --patch_ext ${EXT} --feats_size ${FEATS_SIZE}
	
done

