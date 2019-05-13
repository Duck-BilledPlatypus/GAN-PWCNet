import torch
from torch.autograd import Variable
import itertools
from util.image_pool import ImagePool
import util.task as task
from .base_model import BaseModel
from . import network

class TPModel(BaseModel):
    def name(self):
        return 'TP model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.loss_names = ['img_rec', 'img_G', 'img_D']
        self.visual_names = ['img_s_1', 'img_t_1', 'img_s2t_1', 'img_t2t_1', 'img_s_2', 'img_t_2', 'img_s2t_2', 'img_t2t_2']

        if self.isTrain:
            self.model_names = ['s2t', 'img_D']
        else:
            self.model_names = ['img2task', 's2t']

        # define the transform network
        self.net_s2t = network.define_G(opt.image_nc, opt.image_nc, opt.ngf, opt.transform_layers, opt.norm,
                                                  opt.activation, opt.trans_model_type, opt.init_type, opt.drop_rate,
                                                  False, opt.gpu_ids, opt.U_weight)

        # define the discriminator
        if self.isTrain:
            self.net_img_D = network.define_D(opt.image_nc, opt.ndf, opt.image_D_layers, opt.num_D, opt.norm,
                                              opt.activation, opt.init_type, opt.gpu_ids)
        if self.isTrain:
            self.fake_img_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.l1loss = torch.nn.L1Loss()
            self.nonlinearity = torch.nn.ReLU()
            # initialize optimizers
            self.optimizer_T2Net = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, self.net_s2t.parameters())}],
                                                    lr=opt.lr_trans, betas=(0.5, 0.9))
            self.optimizer_D = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.net_img_D.parameters())),
                                                lr=opt.lr_trans, betas=(0.5, 0.9))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_T2Net)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(network.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)

    def set_input(self, input):
        self.input = input
        self.img_source_1 = input['img_source'][0]
        self.img_target_1 = input['img_target'][0]
        self.img_source_2 = input['img_source'][1]
        self.img_target_2 = input['img_target'][1]

        if len(self.gpu_ids) > 0:
            self.img_source_1 = self.img_source_1.cuda(self.gpu_ids[0], async=True)
            self.img_target_1 = self.img_target_1.cuda(self.gpu_ids[0], async=True)
            self.img_source_2 = self.img_source_2.cuda(self.gpu_ids[0], async=True)
            self.img_target_2 = self.img_target_2.cuda(self.gpu_ids[0], async=True)

    def forward(self):
        self.img_s_1 = Variable(self.img_source_1)
        self.img_t_1 = Variable(self.img_target_1)
        self.img_s_2 = Variable(self.img_source_2)
        self.img_t_2 = Variable(self.img_target_2)
    def backward_D_basic(self, netD, real, fake):

        D_loss = 0
        for (real_i, fake_i) in zip(real, fake):
            # Real
            D_real = netD(real_i.detach())
            # fake
            D_fake = netD(fake_i.detach())

            for (D_real_i, D_fake_i) in zip(D_real, D_fake):
                D_loss += (torch.mean((D_real_i-1.0)**2) + torch.mean((D_fake_i -0.0)**2))*0.5

        D_loss.backward()

        return D_loss

    def backward_D_image(self):
        network._freeze(self.net_s2t)
        network._unfreeze(self.net_img_D)
        size_1 = len(self.img_s2t_1)
        size_2 = len(self.img_s2t_2)
        fake_1 = []
        fake_2 = []
        for i in range(size_1):
            fake_1.append(self.fake_img_pool.query(self.img_s2t_1[i]))
        real_1 = task.scale_pyramid(self.img_t_1, size_1)
        for i in range(size_2):
            fake_2.append(self.fake_img_pool.query(self.img_s2t_2[i]))
        real_2 = task.scale_pyramid(self.img_t_2, size_2)

        self.loss_img_D = self.backward_D_basic(self.net_img_D, real_1, fake_1) + self.backward_D_basic(self.net_img_D, real_2, fake_2)


    def foreward_G_basic(self, net_G, img_s, img_t):

        img = torch.cat([img_s, img_t], 0)
        fake = net_G(img)

        size = len(fake)

        f_s, f_t = fake[0].chunk(2)
        img_fake = fake[1:]

        img_s_fake = []
        img_t_fake = []

        for img_fake_i in img_fake:
            img_s, img_t = img_fake_i.chunk(2)
            img_s_fake.append(img_s)
            img_t_fake.append(img_t)

        return img_s_fake, img_t_fake, f_s, f_t, size

    def backward_synthesis2real(self):

        # image to image transform
        network._freeze(self.net_img_D)
        network._unfreeze(self.net_s2t)

        self.img_s2t_1, self.img_t2t_1, self.img_f_s_1, self.img_f_t_1, size_1 = \
            self.foreward_G_basic(self.net_s2t, self.img_s_1, self.img_t_1)
        self.img_s2t_2, self.img_t2t_2, self.img_f_s_2, self.img_f_t_2, size_2 = \
            self.foreward_G_basic(self.net_s2t, self.img_s_2, self.img_t_2)
        # image GAN loss and reconstruction loss
        img_real_1 = task.scale_pyramid(self.img_t_1, size_1 - 1)
        img_real_2 = task.scale_pyramid(self.img_t_2, size_2 - 1)
        G_loss = 0
        rec_loss = 0

        for i in range(size_1 - 1):
            rec_loss += self.l1loss(self.img_t2t_1[i], img_real_1[i])
            D_fake = self.net_img_D(self.img_s2t_1[i])
            for D_fake_i in D_fake:
                G_loss += torch.mean((D_fake_i - 1.0) ** 2)

        for i in range(size_2 - 1):
            rec_loss += self.l1loss(self.img_t2t_2[i], img_real_2[i])
            D_fake = self.net_img_D(self.img_s2t_2[i])
            for D_fake_i in D_fake:
                G_loss += torch.mean((D_fake_i - 1.0) ** 2)

        self.loss_img_G = G_loss * self.opt.lambda_gan_img
        self.loss_img_rec = rec_loss * self.opt.lambda_rec_img

        total_loss = self.loss_img_G + self.loss_img_rec

        total_loss.backward()

    

    def optimize_parameters(self, epoch_iter):

        self.forward()
        # T2Net
        self.optimizer_T2Net.zero_grad()
        self.backward_synthesis2real()
        self.optimizer_T2Net.step()
        # Discriminator
        self.optimizer_D.zero_grad()
        self.backward_D_image()
        if epoch_iter % 5 == 0:
            self.optimizer_D.step()

