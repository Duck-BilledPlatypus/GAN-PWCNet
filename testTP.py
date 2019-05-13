import os
from options.test_options import TestOptions
import dataloader.data_loader_TPOV_v
from model.models import create_model
import util.util

opt = TestOptions().parse()
dataset = dataloader.data_loader_TPOV_v.dataloader(opt)
dataset_size = len(dataset)


opt.which_epoch = '8'

model = create_model(opt)

save_dir = os.path.join(opt.results_dir,opt.name)
util.util.mkdir(save_dir)
# testing
for j,data in enumerate(dataset):
    model.set_input(data)
    model.test()
    model.save_results(save_dir + '/')