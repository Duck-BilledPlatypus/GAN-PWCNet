#!/bin/bash
GPU_IDs='0'
source activate s2r
python testTPOV.py \
--model testtpov \
--img_target_file_1_v /home/lyc/Desktop/Oxford_1_tpov_v.txt \
--img_target_file_2_v /home/lyc/Desktop/Oxford_2_tpov_v.txt \
--lab_target_file_v /home/lyc/Desktop/Oxflow_tpov_v.txt \
--gpu_ids ${GPU_IDs} \
--batchSize 3 \
--name test_tpov_20190419 \
