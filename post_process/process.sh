#!/bin/bash
DIR='/data3/ian/dsmil-wsi/test-c16/test_data/'
for i in {25..65}
do
	STORE=${DIR}'binary_attention_mask'${i}
	echo  ${STORE}
	mkdir ${STORE}
	python attention_threshold.py ${STORE} ${i}
	python fill_mask.py ${STORE}

done
