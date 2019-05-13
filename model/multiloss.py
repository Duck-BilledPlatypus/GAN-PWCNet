import torch
import torch.nn.functional as F
from torch import nn

def BPE(input_flow, target_flow, tau = [3, 0.05]):
    #calculate bad pixel error for sparse ground truth
    input_flow = input_flow.double()
    target_flow = target_flow.double()
    EPE_map = torch.norm(target_flow - input_flow, 2, 1)
    batch_size = EPE_map.size(0)
    mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)
    mask = ~ mask
    F_mag = torch.norm(target_flow, 2, 1)
    n_err = (mask == 1) & (EPE_map > tau[0]) & (EPE_map / F_mag > tau[1])
    n_err = n_err.sum(2).sum(1)
    n_total = mask.sum(2).sum(1)
    n_err = n_err.double()
    n_total = n_total.double()
    f_err = n_err / n_total
    return f_err.sum() / batch_size
    


def EPE(input_flow, target_flow, sparse=False, mean=True):
    EPE_map = torch.norm(target_flow.double()-input_flow.double(),2,1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum()/batch_size


def sparse_max_pool(input, size):
    '''Downsample the input by considering 0 values as invalid.
    Unfortunately, no generic interpolation mode can resize a sparse map correctly,
    the strategy here is to use max pooling for positive values and "min pooling"
    for negative values, the two results are then summed.
    This technique allows sparsity to be minized, contrary to nearest interpolation,
    which could potentially lose information for isolated data points.'''

    positive = (input > 0).float()
    negative = (input < 0).float()
    output = F.adaptive_max_pool2d(input * positive, size) - F.adaptive_max_pool2d(-input * negative, size)
    return output


def multiscaleEPE(network_output, target_flow, weights=None, sparse=False):
    def one_scale(output, target, sparse):

        b, _, h, w = output.size()

        if sparse:
            target_scaled = sparse_max_pool(target, (h, w))
        else:
            # target_scaled = F.upsample(target, (h, w), mode='bilinear')
            # target_scaled = torch.unsqueeze(target, 0)
            target_scaled = F.upsample(target, (h, w), mode='bilinear', align_corners = False)
            # target_scaled = torch.squeeze(target_scaled, 0)

        return EPE(output, target_scaled, sparse=True, mean=False)
        #return nn.MSELoss()(output, target_scaled)

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article
    assert(len(weights) == len(network_output))

    loss = 0
    for output, weight in zip(network_output, weights):
        loss += weight * one_scale(output, target_flow, sparse)
    return loss


def realEPE(output, target, sparse=False):
    b, _, h, w = target.size()
    upsampled_output = F.upsample(output, (h,w), mode='bilinear', align_corners=False)
    return EPE(upsampled_output, target, sparse, mean=True)