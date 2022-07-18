import os, sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from data_set import *
from assess_quality import *

class ParaLinear(torch.nn.Module):
    def __init__(self):
        super(ParaLinear, self).__init__()
        self.w = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(1, 1)))
        self.b = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(1, 1)))
        self.name = 'linear'
    
    def forward(self, inputs):
        w = torch.exp(self.w)
        return inputs.mul(w) + self.b
    
class ParaSigmoid(torch.nn.Module):
    def __init__(self):
        super(ParaSigmoid, self).__init__()
        self.w = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(1, 1)))
        self.b = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(1, 1)))
        self.linear = ParaLinear()
        self.name = 'sigmoid'
    
    def forward(self, inputs):
        w = torch.exp(self.w)
        return w * torch.sigmoid(self.linear(inputs)) + self.b
        
class ParaPLR(torch.nn.Sequential):    
    def forward(self, inputs):
        outputs = torch.zeros_like(inputs)
        for i, module in enumerate(self._modules.values()):
            outputs += module(inputs)
        return outputs

def bin_count(x):
    bin_counts = []
    last_x = None
    count = 0
    for x_i in x:
        if last_x == None or x_i != last_x:
            if count != 0:
                bin_counts.append(count)
            count = 1
            last_x = x_i
        else:
            count = count + 1
    bin_counts.append(count)
    return torch.Tensor(bin_counts)

def grid_search(x, max_linears, max_sigmoids, bin_ratios, num_epochs, batch_size, config_path):
    def compute_loss(bin_counts):
        return torch.mean(bin_counts) + torch.max(bin_counts) - torch.min(bin_counts)

    num_linears = [i for i in range(1, max_linears + 1)]
    num_sigmoids = [i for i in range(1, max_sigmoids + 1)]
    num_data = len(x)

    best_loss = None
    configs = {}
    
    for num_linear in num_linears:
        for num_sigmoid in num_sigmoids:
            for bin_ratio in bin_ratios:
                num_bin = int(num_data * bin_ratio)
                print('Search # Linears {} # Sigmoids {} # Bins {}'.format(num_linear, num_sigmoid, num_bin))
                epoch_loss = 0
                for epoch in range(num_epochs):
                    funcs = []
                    funcs += [ParaLinear() for i in range(num_linear)]
                    funcs += [ParaSigmoid() for i in range(num_sigmoid)]
                    model = ParaPLR(*funcs).to(device)
                    y = None
                    for i in range(0, num_keys, batch_size):
                        l = i * batch_size
                        r = (i + 1) * batch_size
                        outputs = model(x[l:r])
                        y = torch.cat((y, outputs), 0) if y != None else outputs
                        del outputs
                    
                    bin_size = ((torch.max(y) - torch.min(y)) / num_bin).item()
                    y = torch.floor(y / bin_size)
                    loss = compute_loss(bin_count(y))
                    epoch_loss += loss.item()

                    if best_loss == None or loss < best_loss:
                        best_loss = loss.item()
                        print('Epoch {} Update Best Loss {}'.format(epoch, best_loss))
                        configs['model'] = model.state_dict()
                        configs['num_linear'] = num_linear
                        configs['num_sigmoid'] = num_sigmoid
                        configs['bin_ratio'] = bin_ratio
                        torch.save(configs, config_path)
                    del y
                    del model
                print('Average Loss', epoch_loss / num_epochs)

if __name__ == '__main__':
    input_dir = 'inputs'
    output_dir = 'outputs'
    dataset = 'longlat-200M-100R-zipf'
    workload_path = os.path.join(input_dir, dataset)
    config_path = os.path.join('configs', dataset + '_gs.log')
    device = 'cuda:0'
    batch_size = 131072
    num_epochs = 1000
    normalize = 1e5

    keys = load_data(workload_path, 'training').to(device)
    mean = torch.min(keys)
    var = (torch.max(keys) - torch.min(keys)) / normalize
    keys = (keys - mean) / var
    num_keys = len(keys)

    grid_search(keys, 8, 8, [0.5, 0.7, 0.9], num_epochs, batch_size, config_path)

    configs = torch.load(config_path)
    print('# Linears', configs['num_linear'])
    print('# Sigmoids', configs['num_sigmoid'])
    print('Bin Ratio', configs['bin_ratio'])
    funcs = []
    funcs += [ParaLinear() for i in range(configs['num_linear'])]
    funcs += [ParaSigmoid() for i in range(configs['num_sigmoid'])]
    model = ParaPLR(*funcs).to(device)
    model.load_state_dict(configs['model'])
    num_bin = int(configs['bin_ratio'] * num_keys)
    y = None
    for i in range(0, num_keys, batch_size):
        l = i * batch_size
        r = (i + 1) * batch_size
        outputs = model(keys[l:r])
        y = torch.cat((y, outputs), 0) if y != None else outputs
    bin_size = ((torch.max(y) - torch.min(y)) / num_bin).item()
    y = torch.floor(y / bin_size)
    bin_counts = bin_count(y)
    print('Mean', torch.mean(bin_counts).item())
    print('Max-Min', (torch.max(bin_counts) - torch.min(bin_counts)).item())
