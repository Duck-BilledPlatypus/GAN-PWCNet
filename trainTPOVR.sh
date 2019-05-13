#!/bin/bash
GPU_IDs='1,2,3'
source activate s2r
python trainTPOVR.py --model tpovrmodel \
--img_source_file_1 /home/lyc/Desktop/PFB_1.txt \
--img_target_file_1 /home/lyc/Desktop/Oxford_1_tpovr.txt \
--img_source_file_2 /home/lyc/Desktop/PFB_2.txt \
--img_target_file_2 /home/lyc/Desktop/Oxford_2_tpovr.txt \
--lab_source_file /home/lyc/Desktop/Flow.txt \
--lab_target_file /home/lyc/Desktop/Oxflow_tpovr.txt \
--gpu_ids ${GPU_IDs} \
--batchSize 3 \
--epoch_count 1 \
--display_freq 5 \
--print_freq 5 \
--name tpovr \


