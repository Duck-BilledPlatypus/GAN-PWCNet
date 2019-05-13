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
from PIL import Image
import numpy
import scipy.misc

class TestTPModel(BaseModel):
    def name(self):
        return 'TestTPModel'

    def initialize(self, opt):

        BaseModel.initialize(self, opt)

        self.model_names = ['s2t']

        # define the transform network
        self.net_s2t = network.define_G(opt.image_nc, opt.image_nc, opt.ngf, opt.transform_layers, opt.norm,
                                        opt.activation, opt.trans_model_type, opt.init_type, opt.drop_rate,
                                        False, opt.gpu_ids, opt.U_weight)
        self.load_networks(opt.which_epoch)

    def set_input(self, input):
        self.input = input
        self.img_target_1 = input['img_target'][0]
        self.img_target_2 = input['img_target'][1]

        if len(self.gpu_ids) > 0:
            self.img_target_1 = self.img_target_1.cuda(self.gpu_ids[0], async=True)
            self.img_target_2 = self.img_target_2.cuda(self.gpu_ids[0], async=True)

        self.img_t_1 = Variable(self.img_target_1)
        self.img_t_2 = Variable(self.img_target_2)

    def test(self):

        with torch.no_grad():
            self.img_t2t_1 = self.net_s2t.forward(self.img_t_1)
            self.img_t2t_2 = self.net_s2t.forward(self.img_t_2)
            

    def save_results(self, save_dir):
        batchsize = self.img_t2t_1[-1].size(0)
        img_target_paths = self.input['img_target_paths'][0]

        for i in range(batchsize):
            util.util.mkdir(save_dir + 't2t/' + img_target_paths[i][-17:-9])
            filename = save_dir + 't2t/' + img_target_paths[i][-17:-3] + 'png'
            img_array = self.img_t2t_1[-1][i].data.cpu().numpy().transpose(1,2,0)
            scipy.misc.imsave(filename, img_array)
            # img = Image.fromarray(self.img_t2t_1[-1][i].data.cpu().numpy().transpose(1,2,0).astype(numpy.uint8))
            # img.save(filename)
