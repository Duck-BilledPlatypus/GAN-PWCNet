#!/bin/bash
GPU_IDs='1,2,3'
source activate s2r
python trainTP.py --model tpmodel \
--img_source_file_1 /home/lyc/Desktop/PFB_1.txt \
--img_target_file_1 /home/lyc/Desktop/Oxford_1.txt \
--img_source_file_2 /home/lyc/Desktop/PFB_2.txt \
--img_target_file_2 /home/lyc/Desktop/Oxford_2.txt \
--gpu_ids ${GPU_IDs} \
--batchSize 2