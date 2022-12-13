#!/bin/bash
BAG_PATH=test-c16/patches
MAP_PATH=test-c16/map_paper_init_5ep
EMBEDDER=test-c16/weights/embedder.pth
#EMBEDDER=simsiam/checkpoint_0004.pth.tar
AGGREGATOR=weights/10172022/21.pth
EXT=jpeg
FEATS_SIZE=512
python testing_cts.py --num_classes 1 --bag_path ${BAG_PATH} --map_path ${MAP_PATH} --embedder_weights ${EMBEDDER} --aggregator_weights ${AGGREGATOR} --patch_ext ${EXT} --feats_size ${FEATS_SIZE}
