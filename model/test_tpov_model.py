import torch
from torch.autograd import Variable
from .base_model import BaseModel
from . import network
from util import util
from collections import OrderedDict
from . import multiloss
from .libflow import flowToColor
from flowlib import write_flow, save_flow_image
from . import PWC
import os
import util.util


class TestTPOVModel(BaseModel):
    def name(self):
        return 'TestTPOVModel'

    def initialize(self, opt):
        assert (not opt.isTrain)
        BaseModel.initialize(self, opt)

        self.loss_names_v = ['lab_t']
        self.model_names = ['img2task', 's2t']
        self.loss_lab_t = 0

        # define the transform network
        self.net_s2t = network.define_G(opt.image_nc, opt.image_nc, opt.ngf, opt.transform_layers, opt.norm,
                                        opt.activation, opt.trans_model_type, opt.init_type, opt.drop_rate,
                                        False, opt.gpu_ids, opt.U_weight)
        # define the task network
        if opt.which_epoch == '0':
            pwcnetpath = '/home/lyc/Desktop/PWC-Net/PyTorch/pwc_net.pth.tar'
            self.net_img2task = PWC.pwc_dc_net(pwcnetpath).cuda()
        else:
            self.net_img2task = PWC.pwc_dc_net().cuda()
            self.load_networks(opt.which_epoch)

    def set_input(self, input):
        self.input = input
        self.img_target_1 = input['img_target'][0]
        self.img_target_2 = input['img_target'][1]
        self.lab_target = input['lab_target']

        if len(self.gpu_ids) > 0:
            self.img_target_1 = self.img_target_1.cuda(self.gpu_ids[0], async=True)
            self.img_target_2 = self.img_target_2.cuda(self.gpu_ids[0], async=True)
            self.lab_target = self.lab_target.cuda(self.gpu_ids[0], async=True)

        self.img_t_1 = Variable(self.img_target_1)
        self.img_t_2 = Variable(self.img_target_2)
        self.lab_t = Variable(self.lab_target)

    def test(self):

        with torch.no_grad():
            self.img_t2t_1 = self.net_s2t.forward(self.img_t_1)
            self.img_t2t_2 = self.net_s2t.forward(self.img_t_2)
            self.flow_t, flowlist_t = self.net_img2task.forward(torch.cat([self.img_t2t_1[-1], self.img_t2t_2[-1]], 1))
        self.loss_lab_t += multiloss.BPE(self.flow_t, self.lab_t)


    def test_epoch0(self):
        with torch.no_grad():
            self.flow_t, flowlist_t = self.net_img2task.forward(torch.cat([self.img_t_1, self.img_t_2], 1))
        self.loss_lab_t += multiloss.BPE(self.flow_t, self.lab_t)

    def save_results(self, save_dir):
        batchsize = self.flow_t.size(0)
        img_target_paths = self.input['img_target_paths'][0]
        
        for i in range(batchsize):
            util.util.mkdir(save_dir + 'flow/' + img_target_paths[i][-17:-9])
            util.util.mkdir(save_dir + 'png/' + img_target_paths[i][-17:-9])
            write_flow(self.flow_t[i].data.cpu().numpy().transpose(1,2,0), save_dir + 'flow/'
             + img_target_paths[i][-17:-3] + 'flo')
            save_flow_image(self.flow_t[i].data.cpu().numpy().transpose(1,2,0), save_dir + 'png/'
             + img_target_paths[i][-17:-3] + 'png')

            # save_results