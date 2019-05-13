import os
from options.test_options import TestOptions
import dataloader.data_loader_TPOV_v
from model.models import create_model
import util.util

opt = TestOptions().parse()
bperror = []
dataset = dataloader.data_loader_TPOV_v.dataloader(opt)
dataset_size = len(dataset)

# for i in range(1):
#     opt.which_epoch = str(i)

#     model = create_model(opt)

#     save_dir = os.path.join(opt.results_dir,opt.name, '%s_%s' %(opt.phase, opt.which_epoch))
#     print(save_dir)
#     util.util.mkdir(save_dir)
#     # testing
#     for j,data in enumerate(dataset):
#         print(i,j)
#         model.set_input(data)
#         model.test_epoch0()
#         model.save_results(save_dir + '/')
#     error = model.get_current_errors_v()
#     error['lab_t'] /= dataset_size
#     bperror.append(error['lab_t'])

for i in range(27,30):
    opt.which_epoch = str(i)

    model = create_model(opt)

    save_dir = os.path.join(opt.results_dir,opt.name, '%s_%s' %(opt.phase, opt.which_epoch))
    util.util.mkdir(save_dir)
    # testing
    for j,data in enumerate(dataset):
        print(i,j)
        model.set_input(data)
        model.test()
        model.save_results(save_dir + '/')
    error = model.get_current_errors_v()
    error['lab_t'] /= dataset_size
    bperror.append(error['lab_t'])
print(bperror)