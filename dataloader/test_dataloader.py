from dataloader.data_loader import dataloader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_source_file', type=str, default='/home/lyc/Desktop/datasets/playingforbenchmarks/train/img/008/',
                    help='training and testing dataset for source domain')
parser.add_argument('--img_target_file', type=str, default='/home/lyc/Desktop/Oxford_6000/training/20141216/',
                    help='training and testing dataser for target domain')
parser.add_argument('--lab_source_file', type=str, default='/data/dataset/Image2Depth_SUN_NYU/trainC_SYN10.txt',
                    help='training label for source domain')
parser.add_argument('--lab_target_file', type=str, default='/data/dataset/Image2Depth_SUN_NYU/trainC.txt',
                    help='training label for target domain')
parser.add_argument('--dataset_mode', type=str, default='unpaired',
                    help='chooses how datasets are loaded. [paired| unpaired]')
parser.add_argument('--loadSize', type=list, default=[256, 192],
                    help='load image into same size [256, 192]|[640, 192]')
parser.add_argument('--flip', action='store_true', default=False,
                    help='if specified, do flip the image for data augmentation')
parser.add_argument('--rotation', action='store_true', default=False,
                    help='if specified, rotate the images for data augmentation')
parser.add_argument('--batchSize', type=int, default=6,
                    help='input batch size')
parser.add_argument('--nThreads', type=int, default=8,
                    help='# threads for loading data')
parser.add_argument('--shuffle', action='store_true', default=False,
                    help='if true, takes images randomly')
opt = parser.parse_args()
opt.isTrain = False




dataset = dataloader(opt)
dataset_size = len(dataset) * opt.batchSize
print('training images = %d' % dataset_size)
for i, data in enumerate(dataset):
    print(i)
    print(data)


