#!/bin/bash
BAG_PATH=test-c16/patches
MAP_PATH=test-c16/map_simsiam_m_select_1
EMBEDDER=embedder/C16_paper_simsiam_24ep/embedder.pth
#EMBEDDER=simsiam/checkpoint_0004.pth.tar
AGGREGATOR=weights_C16/10272022/1.pth
EXT=jpeg
FEATS_SIZE=2048
export CUDA_VISIBLE_DEVICES=0,1
python testing_cts_simsiam_vis_critical.py --num_classes 1 --bag_path ${BAG_PATH} --map_path ${MAP_PATH} --embedder_weights ${EMBEDDER} --aggregator_weights ${AGGREGATOR} --patch_ext ${EXT} --feats_size ${FEATS_SIZE}
