#!/bin/bash
base=1
#BAG_PATH=WSI/project_tiff/single/1
#MAP_PATH=WSI/project_tiff/map_1015_more_data

BAG_PATH=test-c16/patches
MAP_PATH=test-c16/map_m_select_1104_

EMBEDDER=embedder/C16_paper_simsiam_24ep/embedder.pth
#EMBEDDER=embedder/project_tiff/embedder.pth
AGGREGATOR=weights/11042022/

EXT=jpeg
FEATS_SIZE=2048

for ((i=0;i<14;i++)); do
	AGGR_WEIGHT=${AGGREGATOR}$((${base}+${i}))".pth"
	DIR=${MAP_PATH}$((${base}+${i}))
	echo ${AGGR_WEIGHT}
	python testing_cts_simsiam.py --num_classes 1 --bag_path ${BAG_PATH} --map_path ${DIR} --embedder_weights ${EMBEDDER} --aggregator_weights ${AGGR_WEIGHT} --patch_ext ${EXT} --feats_size ${FEATS_SIZE}
done

