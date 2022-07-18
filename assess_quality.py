import os, sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from data_set import *

def compute_unordered_keys(keys, verbose=False):
    key_list = keys.squeeze().cpu().detach().numpy().tolist()
    num_unordered = 0
    for i, key in enumerate(key_list):
        if i > 0 and key < key_list[i - 1]:
            # print('Unordered Pair\n%d\t%.32f\n%d\t%.32f' % (i - 1, key_list[i - 1], i, key))
            num_unordered += 1
    del key_list
    return num_unordered


def compute_number_duplicated_keys(keys, verbose=False):
    key_list = keys.squeeze().cpu().detach().numpy().tolist()
    num_duplicated = 0
    for i, key in enumerate(key_list):
        if i > 0 and key == key_list[i - 1]:
            if verbose:
                print(i, key_list[i - 1], key)
            num_duplicated += 1
    del key_list
    return num_duplicated


def build_linear_model(keys, y=None):
    n = keys.shape[0]
    if y == None:
        y = torch.from_numpy(np.arange(n)).unsqueeze(1).to(keys.device)
    elif y == 'prop':
        y = n * (keys - torch.min(keys)) / (torch.max(keys) - torch.min(keys))
    x_sum = keys.sum().item()
    y_sum = y.sum().item()
    xx_sum = (keys * keys).sum().item()
    xy_sum = (keys * y).sum().item()
    if n * xx_sum == x_sum * x_sum:
        slope = 0
    else:
        slope = (n * xy_sum - x_sum * y_sum) / (n * xx_sum - x_sum * x_sum)
    intercept = (y_sum - slope * x_sum) / n
    del y
    return slope, intercept


def compute_conflicts(keys, slope, intercept, amp=-1, tail_percent=0.99, verbose=False):
    n = keys.shape[0]
    intercept = -keys[0].item() * slope + 0.5
    print('Slope {}, intercept {}'.format(slope, intercept))
    pos = (keys * slope + intercept).long().squeeze()
    if amp != -1:
        max_size = int(n * amp)
        pos[pos < 0] = 0
        pos[pos >= max_size] = max_size - 1
    else:
        max_size = pos[-1] + 1
    print('Space amplification {}'.format((pos[-1] - pos[0]) / n))
    conflicts = []
    max_conflicts = 0
    sum_conflicts = 0
    conflicts_per_pos = 0
    last_pos = pos[0]
    for i in range(1, n):
        if pos[i] == pos[i - 1]:
            conflicts_per_pos = conflicts_per_pos + 1
        else:
            last_pos = pos[i]
            max_conflicts = np.maximum(max_conflicts, conflicts_per_pos)
            sum_conflicts = sum_conflicts + conflicts_per_pos
            if conflicts_per_pos > 0:
                conflicts.append(conflicts_per_pos)
            conflicts_per_pos = 0
    if conflicts_per_pos > 0:
        max_conflicts = np.maximum(max_conflicts, conflicts_per_pos)
        sum_conflicts = sum_conflicts + conflicts_per_pos
        conflicts.append(conflicts_per_pos)
    conflicts.sort(key=lambda x: x)
    print(len(conflicts))
    if len(conflicts) > 0:
        tail_conflicts = conflicts[int(tail_percent * len(conflicts)) - 1]
        avg_conflicts = sum_conflicts / len(conflicts)
    else:
        max_conflicts = 0
        tail_conflicts = 0
        avg_conflicts = 0
    del pos
    del conflicts
    return sum_conflicts, max_conflicts, tail_conflicts, avg_conflicts


def evaluate_keys(keys : torch.Tensor, verbose=False):
    print('#' * 100)
    assert keys.shape[1] == 1
    print('Assess the quality of keys', keys.shape[0])
    num_unordered = compute_unordered_keys(keys, verbose=verbose)
    print('Number of unordered keys {}'.format(num_unordered))

    num_duplicated = compute_number_duplicated_keys(keys, verbose=verbose)
    print('Number of duplicated keys {}'.format(num_duplicated))

    slope, intercept = build_linear_model(keys)
    conf_stat = compute_conflicts(keys, slope, intercept, amp=-1, verbose=verbose)
    print('Absolute Conflicts\nTotal conflicts {}, max conflict {}, 99% conflict {}, average conflict {}'.format(conf_stat[0], conf_stat[1], conf_stat[2], conf_stat[3]))

    slope, intercept = build_linear_model(keys)
    conf_stat = compute_conflicts(keys, slope, intercept, amp=1.5, verbose=verbose)
    print('Limited Space Conflicts\nTotal conflicts {}, max conflict {}, 99% conflict {}, average conflict {}'.format(conf_stat[0], conf_stat[1], conf_stat[2], conf_stat[3]))

    slope, intercept = build_linear_model(keys, 'prop')
    conf_stat = compute_conflicts(keys, slope, intercept, amp=1.5, verbose=verbose)
    print('Proportional Building\nTotal conflicts {}, max conflict {}, 99% conflict {}, average conflict {}'.format(conf_stat[0], conf_stat[1], conf_stat[2], conf_stat[3]))
    print('#' * 100)

if __name__ == '__main__':
    workload_path = 'inputs/longlat-200M-100R-zipf'
    keys = load_data(workload_path, 'training')
    evaluate_keys(keys)
    keys, _ = sample_data(keys, 0.1)
    keys = torch.sort(keys, 0)[0]
    n = len(keys)
    fig = plt.figure()
    plt.plot(keys, [i for i in range(len(keys))])
    plt.savefig('origin.png')
    slope, intercept = build_linear_model(keys)
    k = 10
    slope = slope * k / n
    intercept = intercept * k / n
    pos = torch.floor(slope * keys + intercept)
    fig = plt.figure()
    plt.plot(pos, [i for i in range(len(pos))])
    plt.savefig('pos.png')

