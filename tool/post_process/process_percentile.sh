#!/bin/bash
DIR='/data3/ian/dsmil-wsi/test-c16/attention_percentile/'
for i in {1..9}
do
	NUM=99.${i}
	STORE=${DIR}'binary_attention_mask_percentile_99_'${i}
	echo  ${STORE}
	mkdir ${STORE}
	python attention_threshold_percentile.py ${STORE} ${NUM}
	python fill_mask.py ${STORE}

done
