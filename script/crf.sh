#!/bin/bash
#PROB_DIR='/data1/ian/share_code/result_mask_new2/'
RESULT_DIR='/data3/ian/dsmil-wsi/crf1208/'
mkdir ${RESULT_DIR}

for i in {20..45}
do
        PROCESS_STORE=${RESULT_DIR}'crf'${i}
        echo ${PROCESS_STORE}
        mkdir ${PROCESS_STORE}

	python crf.py ${PROCESS_STORE} ${i}

done
~



