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

def sample_data(keys : torch.Tensor, ratio):
    num = np.max((int(keys.shape[0] * ratio), 10000))
    perm = torch.randperm(keys.shape[0])
    index = perm[:num]
    return keys[index]
