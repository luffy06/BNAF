import scipy.stats as ss
import numpy as np
import torch
import os
import psutil

def load_data(location, data_type):
    inputs = np.loadtxt(location + '-' + data_type + '.txt', delimiter='\t', dtype=np.float64, ndmin=2)
    data = torch.from_numpy(inputs).double()
    return data

def seg_data(data, debug=False):
    torch.set_printoptions(precision=10)
    seg_data = []
    seg_times = 100000

    data_list = data.squeeze().numpy().tolist()
    for i, key in enumerate(data_list):
        int_part = int(key)
        float_part = key - int_part
        if float_part == 0.:
            key = key / seg_times
            int_part = int(key)
            float_part = key - int_part
        elif int_part == 0:
            key = key * seg_times
            int_part = int(key)
            float_part = key - int_part
        seg_data.append([int_part, float_part])
    seg_data = torch.Tensor(seg_data)
    return seg_data

def train_data(keys : torch.Tensor, input_dim, train_ratio, grain='point', **kwargs):
    num_train_keys = int(keys.shape[0] * train_ratio)
    if len(keys) < 10000:
        num_train_keys = keys.shape[0]
    emperical_gap = 0
    if grain == 'point':
        perm = torch.randperm(keys.shape[0])
        train_indices = perm[:num_train_keys]
    elif grain == 'batch':
        dim = kwargs['dim']
        num_total_batches = keys.shape[0] // dim
        num_train_batches = num_train_keys // dim
        perm = torch.randperm(num_total_batches)
        train_indices = None
        for i in range(num_train_batches):
            l = perm[i] * dim
            r = (perm[i] + 1) * dim
            indices = torch.arange(l, r)
            train_keys = keys[indices]
            mean_gaps = (train_keys[1:] - train_keys[:-1]).double().mean()
            emperical_gap += mean_gaps
            if train_indices == None:
                train_indices = indices
            else:
                train_indices = torch.cat((train_indices, indices))
        emperical_gap /= num_train_batches
    if input_dim == 1:
        return keys[train_indices], emperical_gap
    elif input_dim == 2:
        return seg_data(keys[train_indices]), emperical_gap
    else:
        return None, None
