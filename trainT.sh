#!/bin/bash
GPU_IDs='0,1,2,3'
source activate s2r
python trainT.py --model tmodel \
--img_source_file /home/lyc/Desktop/PFB_1.txt \
--img_target_file /home/lyc/Desktop/Oxford_1.txt \
--gpu_ids ${GPU_IDs} \
--batchSize 3

