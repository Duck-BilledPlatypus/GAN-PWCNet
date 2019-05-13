import random
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as data
from .image_folder import make_dataset
import torchvision.transforms.functional as F


class CreateDataset(data.Dataset):
    def initialize(self, opt):
        self.opt = opt

        self.img_source_paths_1, self.img_source_size = make_dataset(opt.img_source_file_1)
        self.img_target_paths_1, self.img_target_size = make_dataset(opt.img_target_file_1)
        self.img_source_paths_2, self.img_source_size = make_dataset(opt.img_source_file_2)
        self.img_target_paths_2, self.img_target_size = make_dataset(opt.img_target_file_2)

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
            
            
            img_source_1 = self.transform_augment(img_source_1)
            img_source_2 = self.transform_augment(img_source_2)
            img_target_1 = self.transform_no_augment(img_target_1)
            img_target_2 = self.transform_no_augment(img_target_2)

            return {'img_source': [img_source_1, img_source_2] , 'img_target': [img_target_1, img_target_2],
                    'img_source_paths': [img_source_path_1, img_source_path_2], 'img_target_paths': [img_target_path_1, img_target_path_2]
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
