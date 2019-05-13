
import torch
from torch.autograd import Variable
import itertools
from util.image_pool import ImagePool
import util.task as task
from .base_model import BaseModel
from . import network
from . import PWC
from . import multiloss
from .libflow import flowToColor
import matplotlib.pyplot as plt
import numpy as np
import visdom
from flowlib import write_flow, save_flow_image


class TPOVModel(BaseModel):
    def name(self):
        return 'TPOV model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.vis = visdom.Visdom(port = 8097)
        self.vis1 = self.vis.image(np.random.rand(3, opt.loadSize[1], opt.loadSize[0]),opts=dict(title='sflow'))
        self.vis2 = self.vis.image(np.random.rand(3, opt.loadSize[1], opt.loadSize[0]),opts=dict(title='sgt'))
        self.vis3 = self.vis.image(np.random.rand(3, opt.loadSize[1], opt.loadSize[0]),opts=dict(title='tflow'))
        self.vis4 = self.vis.image(np.random.rand(3, opt.loadSize[1], opt.loadSize[0]),opts=dict(title='tgt'))

        self.loss_names = ['img_rec', 'img_G', 'img_D', 'lab_s']
        self.loss_names_v = ['lab_t_v']
        self.loss_lab_t_v = 0
        self.visual_names = ['img_s_1', 'img_t_1', 'img_s2t_1', 'img_t2t_1', 'img_s_2', 'img_t_2', 'img_s2t_2', 'img_t2t_2']

        if self.isTrain:
            self.model_names = ['s2t', 'img_D', 'img2task']
        else:
            self.model_names = ['img2task', 's2t']

        # define the transform network
        self.net_s2t = network.define_G(opt.image_nc, opt.image_nc, opt.ngf, opt.transform_layers, opt.norm,
                                                  opt.activation, opt.trans_model_type, opt.init_type, opt.drop_rate,
                                                  False, opt.gpu_ids, opt.U_weight)
        pwcnetpath = '/home/lyc/Desktop/PWC-Net/PyTorch/pwc_net.pth.tar'
        self.net_img2task = PWC.pwc_dc_net(pwcnetpath).cuda()
        # self.net_img2task = PWC.pwc_dc_net().cuda()

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
            self.optimizer_T2Net = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, self.net_s2t.parameters())},
                                                     {'params': filter(lambda p: p.requires_grad, self.net_img2task.parameters()),
                                                      'lr': opt.lr_task, 'betas': (0.95, 0.999)}],
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
        self.lab_source = input['lab_source']
        self.lab_target = input['lab_target']



        if len(self.gpu_ids) > 0:
            self.img_source_1 = self.img_source_1.cuda(self.gpu_ids[0], async=True)
            self.img_target_1 = self.img_target_1.cuda(self.gpu_ids[0], async=True)
            self.img_source_2 = self.img_source_2.cuda(self.gpu_ids[0], async=True)
            self.img_target_2 = self.img_target_2.cuda(self.gpu_ids[0], async=True)
            self.lab_source = self.lab_source.cuda(self.gpu_ids[0], async=True)
            self.lab_target = self.lab_target.cuda(self.gpu_ids[0], async=True)

    def forward(self):
        self.img_s_1 = Variable(self.img_source_1)
        self.img_t_1 = Variable(self.img_target_1)
        self.img_s_2 = Variable(self.img_source_2)
        self.img_t_2 = Variable(self.img_target_2)
        self.lab_s = Variable(self.lab_source)
        self.lab_t = Variable(self.lab_target)

    def backward_D_basic(self, netD, real, fake):

        D_loss = 0
        for (real_i, fake_i) in zip(real, fake):
            # Real
            D_real = netD(real_i.detach())
            # fake
            D_fake = netD(fake_i.detach())

            for (D_real_i, D_fake_i) in zip(D_real, D_fake):
                D_loss += (torch.mean((D_real_i-1.0)**2) + torch.mean((D_fake_i -0.0)**2))*0.5


        return D_loss

    def backward_D_image(self):
        network._freeze(self.net_s2t, self.net_img2task)
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

        D_loss_1 = self.backward_D_basic(self.net_img_D, real_1, fake_1)
        D_loss_2 = self.backward_D_basic(self.net_img_D, real_2, fake_2)
        self.loss_img_D = D_loss_1 + D_loss_2
        self.loss_img_D.backward()


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
        network._freeze(self.net_img_D, self.net_img2task)
        network._unfreeze(self.net_s2t)

        self.img_s2t_1, self.img_t2t_1, self.img_f_s_1, self.img_f_t_1, size_1 = \
            self.foreward_G_basic(self.net_s2t, self.img_s_1, self.img_t_1)
        self.img_s2t_2, self.img_t2t_2, self.img_f_s_2, self.img_f_t_2, size_2 = \
            self.foreward_G_basic(self.net_s2t, self.img_s_2, self.img_t_2)


        
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

        total_loss.backward(retain_graph = True)

    def backward_translated2flow(self):

    	network._freeze(self.net_img_D)
        network._unfreeze(self.net_s2t, self.net_img2task)

    	flow_s, flowlist_s = self.net_img2task.forward(torch.cat([self.img_s2t_1[-1], self.img_s2t_2[-1]], 1))
        # print(flow_s)
        # print(self.lab_s)

        flow_t, flowlist_t = self.net_img2task.forward(torch.cat([self.img_t_1, self.img_t_2], 1))
        
        for i in range(1):
            self.img_flow_s = flowToColor(flow_s[i].data.cpu().numpy()).transpose(2,0,1)
            self.vis.images(self.img_flow_s, win = self.vis1)
            self.img_GT_s = flowToColor(self.lab_s[i].data.cpu().numpy()).transpose(2,0,1)
            self.vis.images(self.img_GT_s, win = self.vis2)
            self.img_flow_t = flowToColor(flow_t[i].data.cpu().numpy()).transpose(2,0,1)
            self.vis.images(self.img_flow_t, win = self.vis3)
            self.img_GT_t = flowToColor(self.lab_t[i].data.cpu().numpy()).transpose(2,0,1)
            self.vis.images(self.img_GT_t, win = self.vis4)

    	self.loss_lab_s = multiloss.multiscaleEPE(flowlist_s, self.lab_s/20)


    	total_loss = self.loss_lab_s
    	total_loss.backward()

    def optimize_parameters(self, epoch_iter):

        self.forward()
        # T2Net
        self.optimizer_T2Net.zero_grad()
        self.backward_synthesis2real()
        self.backward_translated2flow()
        self.optimizer_T2Net.step()
        # Discriminator
        self.optimizer_D.zero_grad()
        self.backward_D_image()
        if epoch_iter % 5 == 0:
            self.optimizer_D.step()

    #20190329
    def set_input_v(self, input_v):
        self.input_v = input_v
        self.img_target_1_v = input_v['img_target'][0]
        self.img_target_2_v = input_v['img_target'][1]
        self.lab_target_v = input_v['lab_target']

        if len(self.gpu_ids) > 0:
            self.img_target_1_v = self.img_target_1_v.cuda(self.gpu_ids[0], async=True)
            self.img_target_2_v = self.img_target_2_v.cuda(self.gpu_ids[0], async=True)
            self.lab_target_v = self.lab_target_v.cuda(self.gpu_ids[0], async=True)

        self.img_t_1_v = Variable(self.img_target_1_v)
        self.img_t_2_v = Variable(self.img_target_2_v)
        self.lab_t_v = Variable(self.lab_target_v)
    def validation_target(self):

        with torch.no_grad():
            self.img_t2t_1_v = self.net_s2t.forward(self.img_t_1_v)
            self.img_t2t_2_v = self.net_s2t.forward(self.img_t_2_v)
            flow_t_v, flowlist_t_v = self.net_img2task.forward(torch.cat([self.img_t2t_1_v[-1], self.img_t2t_2_v[-1]], 1))
        self.loss_lab_t_v += multiloss.BPE(flow_t_v, self.lab_t_v)

    def validation_target_epoch0(self):

        with torch.no_grad():
            flow_t_v, flowlist_t_v = self.net_img2task.forward(torch.cat([self.img_t_1_v, self.img_t_2_v], 1))
        self.loss_lab_t_v += multiloss.BPE(flow_t_v, self.lab_t_v)

