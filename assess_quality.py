import os, sys
import torch
import numpy as np
from data_set import *

def compute_unordered_keys(keys, verbose=False):
    assert(keys.shape[1] == 1)
    key_list = keys.squeeze().cpu().numpy().tolist()
    num_unordered = 0
    for i, key in enumerate(key_list):
        if i > 0 and key < key_list[i - 1]:
            num_unordered += 1
    del key_list
    return num_unordered


def compute_number_duplicated_keys(keys, verbose=False):
    assert(keys.shape[1] == 1)
    key_list = keys.squeeze().cpu().numpy().tolist()
    num_duplicated = 0
    for i, key in enumerate(key_list):
        if i > 0 and key == key_list[i - 1]:
            if verbose:
                print(i, key_list[i - 1], key)
            num_duplicated += 1
    del key_list
    return num_duplicated


def build_linear_model(keys):
    n = keys.shape[0]
    y = torch.from_numpy(np.arange(n)).unsqueeze(1)
    x_sum = keys.sum().item()
    y_sum = y.sum().item()
    xx_sum = (keys * keys).sum().item()
    xy_sum = (keys * y).sum().item()
    if n * xx_sum == x_sum * x_sum:
        slope = 0
    else:
        slope = (n * xy_sum - x_sum * y_sum) / (n * xx_sum - x_sum * x_sum)
    intercept = (y_sum - slope * x_sum) / n
    return slope, intercept


def compute_conflicts(keys, amp=-1, tail_percent=0.99, verbose=False):
    n = keys.shape[0]
    slope, intercept = build_linear_model(keys)
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
    if len(conflicts) > 0:
        tail_conflicts = conflicts[int(tail_percent * len(conflicts)) - 1]
        avg_conflicts = sum_conflicts / len(conflicts)
    else:
        max_conflicts = 0
        tail_conflicts = 0
        avg_conflicts = 0

    del pos
    return sum_conflicts, max_conflicts, tail_conflicts, avg_conflicts


def evaluate_keys(keys : torch.Tensor, verbose=False):
    print('#' * 100)
    print('Assess the quality of keys', keys.shape)
    if keys.shape[1] == 1:
        num_unordered = compute_unordered_keys(keys, verbose=verbose)
        print('Number of unordered keys {}'.format(num_unordered))

    num_duplicated = compute_number_duplicated_keys(keys, verbose=verbose)
    print('Number of duplicated keys {}'.format(num_duplicated))

    conf_stat = compute_conflicts(keys, verbose=verbose)
    print('Total conflicts {}, max conflict {}, 99% conflict {}, average conflict {}'.format(conf_stat[0], conf_stat[1], conf_stat[2], conf_stat[3]))
    print('#' * 100)

if __name__ == '__main__':
    workload_path = '../inputs/lognormal-190M-var(1)-100R-zipf'
    keys = load_data(workload_path, 'training')
    evaluate_keys(keys)
