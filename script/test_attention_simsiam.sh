#!/bin/bash
BAG_PATH=test-c16/patches
MAP_PATH=test-c16/map_paper_simsiam_60ep_try_to_reverse
EMBEDDER=embedder/C16_paper_simsiam_ep60_try_to_reproduce/embedder.pth
#EMBEDDER=simsiam/checkpoint_0004.pth.tar
AGGREGATOR=weights/09232022/5.pth
EXT=jpeg
FEATS_SIZE=2048
python test_attention_simsiam.py --num_classes 1 --bag_path ${BAG_PATH} --map_path ${MAP_PATH} --embedder_weights ${EMBEDDER} --aggregator_weights ${AGGREGATOR} --patch_ext ${EXT} --feats_size ${FEATS_SIZE}
