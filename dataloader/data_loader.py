import random
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.transforms.functional as F
from os.path import *
from scipy.misc import imread
from glob import glob
import torch
import numpy as np

class CreateDataset(data.Dataset):
    def initialize(self, opt):
        self.opt = opt

        target_root = opt.img_target_file
        self.target_list = []
        target_folder_list = sorted(glob(join(target_root, '*')))
        for target_folder in target_folder_list:
            target = sorted(glob('/'.join([target_folder,'*.png'])))
            print(target)
            for idx in range(len(target) - 1):
                img_target_1 = target[idx]
                img_target_2 = target[idx + 1]
                if not isfile(img_target_1) or not isfile(img_target_2):
                    continue
                self.target_list += [[img_target_1, img_target_2]]
        self.img_target_size = len(self.target_list)

        if opt.isTrain:
            source_root = opt.img_source_file
            lab_source_root = opt.lab_source_file

            file_list = sorted(glob(join(lab_source_root, '*/*.mat')))
            self.lab_source_list = []
            self.source_list = []

            for file in file_list:
                fbase = file[len(lab_source_root):]
                fprefix = fbase[0:9]
                fnum = int(fbase[9:13])
                # print(fprefix)
                img_source_1 = join(source_root, fprefix + "%d"%(fnum+0) + '.jpg')
                img_source_2 = join(source_root, fprefix + "%d"%(fnum+1) + '.jpg')

                if not isfile(img_source_1) or not isfile(img_source_2) or not isfile(file):
                    continue
                self.source_list += [[img_source_1, img_source_2]]
                self.lab_source_list += [file]
            print(self.lab_source_list)
            print(self.source_list)
            self.img_source_size = len(self.source_list)

    def __getitem__(self, item):
        index = random.randint(0, self.img_target_size - 1)

        img_target_1 = imread(self.target_list[index][0])
        img_target_2 = imread(self.target_list[index][1])

        img_target = [img_target_1, img_target_2]

        img_target = np.array(img_target).transpose(3,0,1,2)
        img_target = torch.from_numpy(img_target.astype(np.float32))

        img_target_path = self.target_list[index]
        if self.opt.isTrain:
            img_source_1 = imread(self.source_list[item % self.img_source_size][0])
            img_source_2 = imread(self.source_list[item % self.img_source_size][1])
            img_source = [img_source_1, img_source_2]
            # lab_source = imread(self.lab_source_list[item % self.img_source_size])

            img_source = np.array(img_source).transpose(3, 0, 1, 2)
            img_source = torch.from_numpy(img_source.astype(np.float32))

            img_source_path = self.source_list[item % self.img_source_size]
            lab_source_path = self.lab_source_list[item % self.img_source_size]

            # return {'img_source': img_source, 'img_target': img_target,
            #         'lab_source': lab_source,
            #         'img_source_paths': img_source_path, 'img_target_paths': img_target_path,
            #         'lab_source_paths': lab_source_path
            #         }
            return {'img_source': img_source, 'img_target': img_target,
                    'img_source_paths': img_source_path, 'img_target_paths': img_target_path,
                    'lab_source_paths': lab_source_path
                    }

        else:
            return {'img_target': img_target,
                    'img_target_paths': img_target_path,
                    }

    def __len__(self):
        if self.opt.isTrain:
            return max(self.img_source_size, self.img_target_size)
        else:
            return self.img_target_size


def dataloader(opt):
    datasets = CreateDataset()
    datasets.initialize(opt)
    dataset = data.DataLoader(datasets, batch_size=opt.batchSize, shuffle=opt.shuffle, num_workers=int(opt.nThreads))
    return dataset

def paired_transform(opt, image, depth):
    scale_rate = 1.0

    if opt.flip:
        n_flip = random.random()
        if n_flip > 0.5:
            image = F.hflip(image)
            depth = F.hflip(depth)

    if opt.rotation:
        n_rotation = random.random()
        if n_rotation > 0.5:
            degree = random.randrange(-500, 500)/100
            image = F.rotate(image, degree, Image.BICUBIC)
            depth = F.rotate(depth, degree, Image.BILINEAR)

    return image, depth, scale_rate


def get_transform(opt, augment):
    transforms_list = []

    if augment:
        if opt.isTrain:
            transforms_list.append(transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0))




    transforms_list += [
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    return transforms.Compose(transforms_list)
