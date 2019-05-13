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

        self.img_source_paths_1, self.img_source_size = make_dataset(opt.img_source_file_1)
        self.img_target_paths_1, self.img_target_size = make_dataset(opt.img_target_file_1)
        self.img_source_paths_2, self.img_source_size = make_dataset(opt.img_source_file_2)
        self.img_target_paths_2, self.img_target_size = make_dataset(opt.img_target_file_2)

        self.lab_source_paths, self.lab_source_size = make_dataset(opt.lab_source_file)
        self.lab_target_paths, self.lab_target_size = make_dataset(opt.lab_target_file)


        self.transform_augment = get_transform(opt, True)
        self.transform_no_augment = get_transform(opt, False)

    def __getitem__(self, item):
        index = random.randint(0, self.img_target_size - 1)
        img_source_path_1 = self.img_source_paths_1[item % self.img_source_size]
        img_source_path_2 = self.img_source_paths_2[item % self.img_source_size]
        if self.opt.dataset_mode == 'paired':
            img_target_path_1 = self.img_target_paths_1[item % self.img_target_size]
            img_target_path_2 = self.img_target_paths_2[item % self.img_target_size]
        elif self.opt.dataset_mode == 'unpaired':
            img_target_path_1 = self.img_target_paths_1[index]
            img_target_path_2 = self.img_target_paths_2[index]
        else:
            raise ValueError('Data mode [%s] is not recognized' % self.opt.dataset_mode)

        img_source_1 = Image.open(img_source_path_1).convert('RGB')
        img_source_2 = Image.open(img_source_path_2).convert('RGB')
        img_target_1 = Image.open(img_target_path_1).convert('RGB')
        img_target_2 = Image.open(img_target_path_2).convert('RGB')
        img_source_1 = img_source_1.resize([self.opt.loadSize[0], self.opt.loadSize[1]], Image.BICUBIC)
        img_source_2 = img_source_2.resize([self.opt.loadSize[0], self.opt.loadSize[1]], Image.BICUBIC)
        img_target_1 = img_target_1.resize([self.opt.loadSize[0], self.opt.loadSize[1]], Image.BICUBIC)
        img_target_2 = img_target_2.resize([self.opt.loadSize[0], self.opt.loadSize[1]], Image.BICUBIC)

        if self.opt.isTrain:
            
            lab_source_path = self.lab_source_paths[item % self.lab_source_size]
            # print(lab_source_path)
            lab_source = read_flo_file(lab_source_path)
            # flowtool = flow2img.Flow()
            # flow = flowtool._readFlow(lab_source_path)
            # print('flow:', flow.shape)
            # img = flowtool._flowToColor(lab_source.transpose(2,0,1))
            # img = Image.fromarray(img).save('/home/lyc/Desktop/flow.png')
            lab_source_h, lab_source_w = lab_source.shape[0], lab_source.shape[1]
        
            lab_source = resample(lab_source, [self.opt.loadSize[1], self.opt.loadSize[0]])
            # print(lab_source.shape)
            lab_source = torch.from_numpy(lab_source.transpose(2,0,1))
            lab_source[0] = lab_source[0] * self.opt.loadSize[0] / lab_source_w
            lab_source[1] = lab_source[1] * self.opt.loadSize[1] / lab_source_h



            lab_target_path =self.lab_target_paths[item % self.lab_target_size]

            lab_target = read_flo_file(lab_target_path)

            lab_target_h, lab_target_w = lab_target.shape[0], lab_target.shape[1]

            lab_target = resample(lab_target, [self.opt.loadSize[1], self.opt.loadSize[0]])

            lab_target = torch.from_numpy(lab_target.transpose(2,0,1))
            lab_target[0] = lab_target[0] * self.opt.loadSize[0] / lab_target_w
            lab_target[1] = lab_target[1] * self.opt.loadSize[1] / lab_target_h

            # print('lab_source:',lab_source.size())

            img_source_1 = self.transform_augment(img_source_1)
            img_source_2 = self.transform_augment(img_source_2)
            img_target_1 = self.transform_no_augment(img_target_1)
            img_target_2 = self.transform_no_augment(img_target_2)
            # print('img_source:',img_source_1.size())

            return {'img_source': [img_source_1, img_source_2] , 'img_target': [img_target_1, img_target_2],
                    'lab_source': lab_source, 
                    'lab_target': lab_target, 
                    'img_source_paths': [img_source_path_1, img_source_path_2], 'img_target_paths': [img_target_path_1, img_target_path_2],
                    'lab_source_path': lab_source_path,
                    'lab_target_path': lab_target_path
                    }

        else:
            img_source = self.transform_augment(img_source)
            img_target = self.transform_no_augment(img_target)
            return {'img_source': img_source, 'img_target': img_target,
                    'img_source_paths': img_source_path, 'img_target_paths': img_target_path,

                    }

    def __len__(self):
        return max(self.img_source_size, self.img_target_size)

    def name(self):
        return 'T^2Dataset'


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
