#!/bin/bash
#seed="64 921 911 77 1894 52 36 484 1 2 5 487 55 44 18 99 87"
export CUDA_VISIBLE_DEVICES=0,1
N_CRITICAL=1

#: << EOF
seed="77 52"
#seed="64 921 911 77 1894 52 36 484 1 2 5 487 55 44 18 99 87"
#seed="77"

#DATASET=C16_paper_simsiam_24ep
DATASET=project_tiff
FEATS_SIZE=2048
LR=0.0002
EP=2
GPU_ID='0 1'
SPLIT=0.2
#WEIGHT='weights_C16/10262022/1.pth'

for i in $seed; do
	python train_tcga.py --dataset=${DATASET} --num_classes 1 --feats_size ${FEATS_SIZE} \
        	             --lr ${LR} --num_epochs ${EP} --gpu_index ${GPU_ID} --split ${SPLIT} --seed ${i}  --n_critical ${N_CRITICAL}
                	     #--aggregator_weights ${WEIGHT}   
done
#EOF

#: << EOF
base=5
#BAG_PATH=test-c16/patches
#MAP_PATH=test-c16/map_m_select_1106_
BAG_PATH=WSI/project_tiff/single/1
MAP_PATH=WSI/project_tiff/map_m_select_1106_
#EMBEDDER=embedder/C16_paper_simsiam_24ep/embedder.pth
EMBEDDER=embedder/project_tiff/embedder.pth
#EMBEDDER=embedder/C16_paper_simsiam_24ep/embedder.pth
AGGREGATOR=weights_C16/11062022/
EXT=jpeg
FEATS_SIZE=2048

for ((i=0;i<2;i++)); do
	AGGR_WEIGHT=${AGGREGATOR}$((${base}+${i}))".pth"
	DIR=${MAP_PATH}$((${base}+${i}))
	echo ${AGGR_WEIGHT}
	python testing_cts_simsiam_vis_critical.py --num_classes 1 --bag_path ${BAG_PATH} --map_path ${DIR} --embedder_weights ${EMBEDDER} --aggregator_weights ${AGGR_WEIGHT} --patch_ext ${EXT} --feats_size ${FEATS_SIZE} --n_critical ${N_CRITICAL}
done
#EOF
