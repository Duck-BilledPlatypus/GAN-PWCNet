import time
from options.train_options import TrainOptions
import dataloader.data_loader_TPOV
import dataloader.data_loader_TPOV_v

from model.models import create_model
from util.visualizer import Visualizer

opt = TrainOptions().parse()
opt_v = TrainOptions().parse()
opt_v.batchSize = 3

dataset = dataloader.data_loader_TPOV.dataloader(opt)
dataset_v = dataloader.data_loader_TPOV_v.dataloader(opt_v)

dataset_size = len(dataset) * opt.batchSize
dataset_size_v = len(dataset_v)
dataset_size_v = float(dataset_size_v)
print('training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
visualizer_v = Visualizer(opt_v)
total_steps=0
vloss = 0

#epoch 0
for i, data in enumerate(dataset_v):
    # print(data.keys())
    print('validation count:', i)
    model.set_input_v(data)
    model.validation_target_epoch0()
errors_v = model.get_current_errors_v()
vloss += errors_v['lab_t_v']
errors_v['lab_t_v'] /= dataset_size_v
visualizer_v.plot_current_errors_v(0, opt_v, errors_v)


for epoch in range(opt.epoch_count, opt.niter+opt.niter_decay+1):
    epoch_start_time = time.time()
    epoch_iter = 0


    # training
    for i, data in enumerate(dataset):

        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters(i)

        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.save_networks('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch (epoch %d, iters %d)' % (epoch, total_steps))
        model.save_networks('latest')
        model.save_networks(epoch)

    print ('End of the epoch %d / %d \t Time Take: %d sec' %
           (epoch, opt.niter + opt.niter_decay, time.time()-epoch_start_time))

    # validation
    for i, data in enumerate(dataset_v):
        model.set_input_v(data)
        model.validation_target()
    errors_v = model.get_current_errors_v()
    errors_v['lab_t_v'] -= vloss
    vloss += errors_v['lab_t_v']
    errors_v['lab_t_v'] /= dataset_size_v
    visualizer_v.plot_current_errors_v(epoch, opt_v, errors_v)

    model.update_learning_rate()
