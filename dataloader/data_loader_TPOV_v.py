import random
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as data
from .image_folder import make_dataset
import torchvision.transforms.functional as F
from flowlib import read_flo_file, resample, show_flow, flow_to_image
import torch
import matplotlib.pyplot as plt
import flow2img
from PIL import Image

class CreateDataset(data.Dataset):
    def initialize(self, opt):
        self.opt = opt

        self.img_target_paths_1, self.img_target_size = make_dataset(opt.img_target_file_1_v)
        self.img_target_paths_2, self.img_target_size = make_dataset(opt.img_target_file_2_v)
        self.lab_target_paths, self.lab_target_size = make_dataset(opt.lab_target_file_v)

        self.transform_augment = get_transform(opt, True)
        self.transform_no_augment = get_transform(opt, False)

    def __getitem__(self, item):
        img_target_path_1 = self.img_target_paths_1[item]
        img_target_path_2 = self.img_target_paths_2[item]
    
        img_target_1 = Image.open(img_target_path_1).convert('RGB')
        img_target_2 = Image.open(img_target_path_2).convert('RGB')

        img_target_1 = img_target_1.resize([self.opt.loadSize_v[0], self.opt.loadSize_v[1]], Image.BICUBIC)
        img_target_2 = img_target_2.resize([self.opt.loadSize_v[0], self.opt.loadSize_v[1]], Image.BICUBIC)
        
        lab_target_path =self.lab_target_paths[item]
        lab_target = read_flo_file(lab_target_path)
        lab_target_h, lab_target_w = lab_target.shape[0], lab_target.shape[1]
        lab_target = resample(lab_target, [self.opt.loadSize_v[1], self.opt.loadSize_v[0]])
        lab_target = torch.from_numpy(lab_target.transpose(2,0,1))
        lab_target[0] = lab_target[0] * self.opt.loadSize_v[0] / lab_target_w
        lab_target[1] = lab_target[1] * self.opt.loadSize_v[1] / lab_target_h

        img_target_1 = self.transform_no_augment(img_target_1)
        img_target_2 = self.transform_no_augment(img_target_2)
            # print('img_source:',img_source_1.size())

        return {'img_target': [img_target_1, img_target_2],
                'lab_target': lab_target, 
                'img_target_paths': [img_target_path_1, img_target_path_2],
                'lab_target_path': lab_target_path
                }

    def __len__(self):
        return self.img_target_size

    def name(self):
        return 'Validation Dataset'


def dataloader(opt):
    datasets = CreateDataset()
    datasets.initialize(opt)
    dataset = data.DataLoader(datasets, batch_size=opt.batchSize, shuffle=opt.shuffle, num_workers=int(opt.nThreads))
    return dataset

def get_transform(opt, augment):
    transforms_list = []

    if augment:
        if opt.isTrain:
            transforms_list.append(transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0))
    transforms_list += [
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    return transforms.Compose(transforms_list)
