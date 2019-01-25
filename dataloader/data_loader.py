import random
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as data
# from torchvision.transforms import functional as F
import torchvision.transforms.functional as F
from os.path import *
from scipy.misc import imread
from glob import glob
import torch
import numpy as np
from torch.autograd import Variable
from torch.nn.functional import upsample

class CreateDataset(data.Dataset):
    def initialize(self, opt):
        self.opt = opt

        target_root = opt.img_target_file
        self.target_list = []
        target_folder_list = sorted(glob(join(target_root, '*')))
        for target_folder in target_folder_list:
            target = sorted(glob('/'.join([target_folder,'*.png'])))

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

            file_list = sorted(glob(join(lab_source_root, '*/*.flo')))
            self.lab_source_list = []
            self.source_list = []

            for file in file_list:
                fbase = file[len(lab_source_root):]
                fprefix = fbase[0:9]
                fnum = int(fbase[9:13])

                img_source_1 = join(source_root, fprefix + "%d"%(fnum+0) + '.jpg')
                img_source_2 = join(source_root, fprefix + "%d"%(fnum+1) + '.jpg')

                if not isfile(img_source_1) or not isfile(img_source_2) or not isfile(file):
                    continue
                self.source_list += [[img_source_1, img_source_2]]
                self.lab_source_list += [file]

            self.img_source_size = len(self.source_list)


    def __getitem__(self, item):
        index = random.randint(0, self.img_target_size - 1)

        img_target_1 = Image.open(self.target_list[index][0]).convert('RGB')
        img_target_2 = Image.open(self.target_list[index][1]).convert('RGB')
        img_target_1 = resize_normal(self.opt,img_target_1)
        img_target_2 = resize_normal(self.opt,img_target_2)
        img_target = torch.cat([img_target_1,img_target_2], 0)


        img_target_path = self.target_list[index]
        if self.opt.isTrain:
            img_source_1 = Image.open(self.source_list[item % self.img_source_size][0])
            img_source_2 = Image.open(self.source_list[item % self.img_source_size][1])
            img_source_1 = resize_normal(self.opt,img_source_1)
            img_source_2 = resize_normal(self.opt,img_source_2)
            # print('img_source_2:', img_source_2.)
            img_source = torch.cat([img_source_1, img_source_2], 0)


            img_source_path = self.source_list[item % self.img_source_size]
            lab_source_path = self.lab_source_list[item % self.img_source_size]


            lab_source = read_flo_file(lab_source_path).transpose(2,0,1)
            # print('lab_source_original:',lab_source.shape)
            # lab_source = lab_source[:,:,:]
            lab_source = torch.unsqueeze(torch.from_numpy(lab_source), 0)
            # print(lab_source.size())
            lab_source = upsample(lab_source,(self.opt.loadSize[1],self.opt.loadSize[0]), mode='bilinear')
            # lab_source = upsample(lab_source, (540,960), mode='bilinear')
            # print('lab_source:',lab_source.size())
            lab_source = torch.squeeze(lab_source, 0)
            lab_source_u = lab_source[0, :, :]
            lab_source_v = lab_source[1,:,:]
            lab_source_u = lab_source_u/(1920/640)
            lab_source_v = lab_source_v/(1080/384)
            lab_source[0,:,:] = lab_source_u
            lab_source[1,:,:] = lab_source_v
            # print(lab_source.size())


            # return {'img_source': img_source, 'img_target': img_target,
            #         'lab_source': lab_source,
            #         'img_source_paths': img_source_path, 'img_target_paths': img_target_path,
            #         'lab_source_paths': lab_source_path
            #         }
            return {'img_source': img_source, 'img_target': img_target,
                    'lab_source': lab_source,
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

def read_flo_file(filename):
    """
    Read from Middlebury .flo file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:

        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h[0], w[0], 2))
        f.close()
    return data2d


def resize_flow(flow, h, w, opt):
    flow = Variable(flow).cuda(opt.gpu)
    bs_, c_, h_, w_ = flow.size()
    hratio = float(h) / float(h_)
    wratio = float(w) / float(w_)
    flow_out = F.upsample(flow, (h, w), mode='bilinear', align_corners=False)
    flow_out[:,0,:,:] = flow_out[:,0, :,:] * wratio
    flow_out[:,1,:,:] = flow_out[:,1,:,:] * hratio
    return flow_out

def resize_normal(opt, img):
    img = img.resize([opt.loadSize[0],opt.loadSize[1]], Image.BICUBIC)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
    return img

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
