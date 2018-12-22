from dataloader.data_loader import dataloader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_source_file', type=str, default='/home/lyc/Desktop/datasets/playingforbenchmarks/train/img/',
                    help='training and testing dataset for source domain')
parser.add_argument('--img_target_file', type=str, default='/home/lyc/Desktop/Oxford_6000/training/',
                    help='training and testing dataser for target domain')
parser.add_argument('--lab_source_file', type=str, default='/home/lyc/Desktop/datasets/playingforbenchmarks/gt/train/flow/',
                    help='training label for source domain')
parser.add_argument('--dataset_mode', type=str, default='unpaired',
                    help='chooses how datasets are loaded. [paired| unpaired]')
parser.add_argument('--loadSize', type=list, default=[256, 192],
                    help='load image into same size [256, 192]|[640, 192]')
parser.add_argument('--flip', action='store_true', default=False,
                    help='if specified, do flip the image for data augmentation')
parser.add_argument('--rotation', action='store_true', default=False,
                    help='if specified, rotate the images for data augmentation')
parser.add_argument('--batchSize', type=int, default=1,
                    help='input batch size')
parser.add_argument('--nThreads', type=int, default=8,
                    help='# threads for loading data')
parser.add_argument('--shuffle', action='store_true', default=False,
                    help='if true, takes images randomly')
opt = parser.parse_args()
opt.isTrain = True




dataset = dataloader(opt)
print(len(dataset))
dataset_size = len(dataset) * opt.batchSize
print('training images = %d' % dataset_size)

for i, data in enumerate(dataset):
    print(i)
    print(data['img_target'].size())

#
# from os.path import *
# from glob import glob
# target_root = '/home/lyc/Desktop/Oxford_6000/training/'
# folder_list = sorted(glob(join(target_root,'*')))
# for folder in folder_list[0:-1]:
#     print(folder)