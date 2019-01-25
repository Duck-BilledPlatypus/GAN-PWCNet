#!/bin/bash
GPU_IDs='1,2,3'
source activate s2r
#CUDA_VISIBLE_DEVICES=${GPU_IDs} 
python trainTransform.py --model transform \
--img_source_file /home/lyc/Desktop/datasets/playingforbenchmarks/train/img/ \
--img_target_file /home/lyc/Desktop/Oxford_6000/training/ \
--lab_source_file /home/lyc/Desktop/datasets/playingforbenchmarks/train/flow/ \
--gpu_ids ${GPU_IDs}
