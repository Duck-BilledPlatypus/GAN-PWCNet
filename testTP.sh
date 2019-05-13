#!/bin/bash
GPU_IDs='3'
source activate s2r
python testTP.py \
--model testtp \
--img_target_file_1_v /home/lyc/Desktop/Oxford_1_tpov_v.txt \
--img_target_file_2_v /home/lyc/Desktop/Oxford_2_tpov_v.txt \
--lab_target_file_v /home/lyc/Desktop/Oxflow_tpov_v.txt \
--gpu_ids ${GPU_IDs} \
--batchSize 4 \
--name test_tpov_20190401 \
