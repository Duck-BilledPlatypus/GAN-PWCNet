#!/bin/bash
GPU_IDs='1,2,3'
source activate s2r
python trainTPOV.py --model tpovmodel \
--img_source_file_1 /home/lyc/Desktop/syn_1_tpov_v3.txt \
--img_target_file_1 /home/lyc/Desktop/Oxford_1_tpovr.txt \
--img_source_file_2 /home/lyc/Desktop/syn_2_tpov_v3.txt \
--img_target_file_2 /home/lyc/Desktop/Oxford_2_tpovr.txt \
--lab_source_file /home/lyc/Desktop/syn_f_tpov_v3.txt \
--lab_target_file /home/lyc/Desktop/Oxflow_tpovr.txt \
--img_target_file_1_v /home/lyc/Desktop/Oxford_1_tpov_v.txt \
--img_target_file_2_v /home/lyc/Desktop/Oxford_2_tpov_v.txt \
--lab_target_file_v /home/lyc/Desktop/Oxflow_tpov_v.txt \
--gpu_ids ${GPU_IDs} \
--batchSize 3 \
--epoch_count 1 \
--display_freq 1 \
--print_freq  20 \
--name tpov20190419 \
