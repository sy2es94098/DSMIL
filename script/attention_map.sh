#!/bin/bash

BAG_PATH=test-c16/patches
MAP_PATH=test-c16/output
EMBEDDER=embedder/C16_paper_simsiam_24ep/embedder.pth
AGGREGATOR=weights_C16/10262022/
EXT=jpeg
FEATS_SIZE=2048
NORM='i'

python attention_map.py --num_classes 1 --bag_path ${BAG_PATH} --map_path ${DIR} --embedder_weights ${EMBEDDER} --aggregator_weights ${AGGR_WEIGHT} --patch_ext ${EXT} --feats_size ${FEATS_SIZE} --norm ${NORM}
	
