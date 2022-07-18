import os
import json
import argparse
import pprint
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import gc
import objgraph
import psutil
import torch
import time
import sys
import random
from tqdm import trange
from bnaf import *
from data_set import *
from assess_quality import *
from data.generate2d import sample2d, energy2d

torch.set_default_dtype(torch.float64)

def create_model(args, use_tanh=True, verbose=False):
    
    flows = [Encoder(args.input_dim, args.shifts)] if args.input_dim > 1 else []
    for f in range(args.flows):
        # First layer
        layers = [MaskedWeight(args.input_dim, 
                                args.input_dim * args.hidden_dim, 
                                dim=args.input_dim, bias=False, 
                                all_pos=args.all_pos, limited_exp=args.limited_exp), 
                    Tanh() if use_tanh else ReLU()]
        # Inner layers
        for _ in range(args.layers - 1):
            layers.append(MaskedWeight(args.input_dim * args.hidden_dim, 
                                        args.input_dim * args.hidden_dim, 
                                        dim=args.input_dim, bias=False, 
                                        all_pos=args.all_pos, limited_exp=args.limited_exp))
            layers.append(Tanh() if use_tanh else ReLU())
        # Last layer
        layers += [MaskedWeight(args.input_dim * args.hidden_dim, 
                                args.input_dim, 
                                dim=args.input_dim, bias=False, 
                                all_pos=args.all_pos, limited_exp=args.limited_exp)]
        flows.append(
            BNAF(*(layers), res=False)
        )
        if f < args.flows - 1:
            flows.append(Permutation(args.input_dim, 'flip'))

    if args.input_dim > 1:
        flows.append(Decoder(args.input_dim, args.de_type, args.trim))

    model = Sequential(*flows).to(args.device)
    
    if verbose:
        print('{}'.format(model))
        print('Parameters={}, n_dims={}'.format(sum((p != 0).sum() 
                                                    if len(p.shape) > 1 else torch.tensor(p.shape).item() 
                                                    for p in model.parameters()), args.input_dim))
    return model


def compute_log_p_x(model, x_mb, args):
    eps = 1e10 ** -5
    y_mb, log_diag_j_mb = model(x_mb)
    if torch.isnan(y_mb).any():
        y_mb = torch.nan_to_num(y_mb, nan=eps)
    mean = torch.zeros_like(y_mb)
    var = torch.ones_like(y_mb)
    if args.de_type == 'trim':
        var[:, args.trim] *= 1e50
    log_p_y_mb = torch.distributions.Normal(mean, var).log_prob(y_mb).sum(-1)
    return log_p_y_mb + log_diag_j_mb


def train_density1d(data_loader, model, optimizer, scheduler, best_loss, args):
    model.train()
    for epoch in range(args.steps):
        print('Epoch:{}'.format(epoch))
        start_time = time.time()
        epoch_loss = 0
        for i, x in enumerate(data_loader):
            x_mb = x[0].to(args.device)
            loss = -compute_log_p_x(model, x_mb, args).mean()
            if np.isnan(loss.item()):
                print('Loss is NaN')
                return False
            epoch_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                            max_norm=args.clip_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(loss)
            if print_detail:
                break
        print('Loss:{}'.format(epoch_loss / len(data_loader)))
        if np.isinf(epoch_loss):
            print('Loss is INF')
            return None
        if np.fabs(best_loss - epoch_loss) < 0.01 * np.fabs(best_loss):
            print('Early stop at epoch-{}'.format(epoch))
            break
        if epoch_loss < best_loss:
            # print('Saving model at epoch-{}'.format(epoch))
            best_loss = epoch_loss
        if print_detail:
            break
        train_durations = time.time() - start_time
        print('Time Cost of Epoch {}'.format(train_durations))
    return best_loss


def test_density1d(data_loader, model, args):
    model.eval()
    y = None
    with torch.no_grad():
        for i, x in enumerate(data_loader):
            x_mb = x[0].to(args.device)
            if print_detail:
                print_tensor(x_mb, 'Inputs')
            y_mb = model(x_mb)
            if args.de_type == 'trim':
                y_mb = y_mb[:, args.trim].unsqueeze(1)
            if print_detail:
                print_tensor(y_mb, 'Outputs')
            if torch.isnan(y_mb).any():
                print('NAN KEYS')
                exit()
            y = torch.cat((y, y_mb), 0) if y != None else y_mb
            del y_mb
            if print_detail:
                break
        return y


def load(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def save(model, optimizer, path):
    d = {}
    d["model"] = model.state_dict()
    d["optimizer"] = optimizer.state_dict()
    torch.save(d, path)


def save_weights(dataset, res_dir, output_filename, model, mean, var, args):
    weight_output_file = os.path.join(res_dir, output_filename + '-weights.txt')
    f = open(weight_output_file, 'w')
    model.eval()
    f.write('%d\t%d\t%d\n' % (args.input_dim, args.input_dim * args.hidden_dim, args.layers + 1))
    f.write('%.16f\t%.16f\n' % (mean.item(), var.item()))
    model.save_weights(f)
    f.close()

def transform_workload(keys, args, model, save_path):
    batch_dim = int(np.minimum(keys.shape[0], args.batch_dim))
    data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(keys), batch_size=batch_dim, num_workers=16)
    tran_keys = test_density1d(data_loader, model, args).squeeze().unsqueeze(1).cpu()
    return tran_keys


def plot_dist(keys, path):
    x = torch.sort(keys, 0)[0]
    fig = plt.figure()
    # plt.hist(x.numpy().tolist(), bins=30, density=True)
    plt.plot(x.numpy().tolist())
    plt.savefig(path)
    plt.close()


def run_bnaf(args):
    output_filename = args.dataset
    stdout_filename = os.path.join(args.output_dir, output_filename +'_results.log')
    sys.stdout = open(stdout_filename, 'w')

    if (args.save or args.savefig) and not args.load:
        print('Process: Creating directory experiment..')
        os.mkdir(args.path)
        with open(os.path.join(args.path, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=4, sort_keys=True)
    
    print('Arguments')
    pprint.pprint(args.__dict__)

    # Loading data, analyzing data, segmenting data
    print('Process: Loading data...')
    start_time = time.time()
    load_keys = load_data(os.path.join(args.input_dir, args.dataset), 'training')
    load_durations = time.time() - start_time
    print('Time Cost of Loading data {}'.format(load_durations))

    # print('Process: Evaluating original keys...')
    # evaluate_keys(load_keys)

    mean = torch.Tensor([0])
    var = torch.Tensor([1])
    if args.input_dim > 1:
        mean = torch.min(load_keys)
        var = (torch.max(load_keys) - torch.min(load_keys)) / args.shifts
        load_keys = (load_keys - mean) / var

    data_loader_list = []
    for i in range(args.num_train):
        print('Process: Spliting training data...')
        train_keys = sample_data(load_keys, args.train_ratio)
        print('Number of training keys {}'.format(train_keys.shape[0]))

        # if args.plot != False:
        #     print('Process: Plot distribution...')
        #     plot_dist(train_keys, os.path.join(args.output_dir, output_filename + '_' + str(i) + '_orgin_dist.png'))
        #     plot_dist(torch.floor(train_keys), os.path.join(args.output_dir, output_filename + '_' + str(i) + '_floor_dist.png'))
        #     plot_dist(train_keys - torch.floor(train_keys), os.path.join(args.output_dir, output_filename + '_' + str(i) + '_res_dist.png'))

        batch_dim = int(np.minimum(train_keys.shape[0], args.batch_dim))
        data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_keys), batch_size=batch_dim, num_workers=16)
        data_loader_list.append(data_loader)

    model = create_model(args, verbose=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

    if args.load:
        load(model, optimizer, os.path.join(args.load, 'checkpoint.pt'))
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.decay,
                                                                patience=args.patience,
                                                                min_lr=5e-4, verbose=True,
                                                                threshold_mode='abs')
        print('Process: Training...')
        start_time = time.time()
        best_loss = 1e20
        for i in range(args.num_train):
            # print('Process: Spliting training data...')
            # train_keys = sample_data(load_keys, args.train_ratio)
            # print('Number of training keys {}'.format(train_keys.shape[0]))

            # if args.plot != False:
            #     print('Process: Plot distribution...')
            #     plot_dist(train_keys, os.path.join(args.output_dir, output_filename + '_' + str(i) + '_orgin_dist.png'))
            #     plot_dist(torch.floor(train_keys), os.path.join(args.output_dir, output_filename + '_' + str(i) + '_floor_dist.png'))
            #     plot_dist(train_keys - torch.floor(train_keys), os.path.join(args.output_dir, output_filename + '_' + str(i) + '_res_dist.png'))

            data_loader = data_loader_list[i]
            best_loss = train_density1d(data_loader, model, optimizer, scheduler, best_loss, args)
            if best_loss == None:
                return 
            # del train_keys
        train_durations = time.time() - start_time
        print('Time Cost of Training {}'.format(train_durations))
        # save(model, optimizer, os.path.join(args.load or args.path, 'checkpoint.pt'))

    print('Process: Saving weights...')
    save_weights(args.dataset, args.output_dir, output_filename, model, mean, var, args)
    
    print('Process: Transforming keys...')
    output_training_path = os.path.join(args.output_dir, output_filename + '-training-bnaf.txt')
    tran_keys = transform_workload(load_keys, args, model, output_training_path)
    del load_keys
    sort = False
    last_order = None
    for i in range(tran_keys.shape[0]):
        if i > 0 and tran_keys[i].item() != tran_keys[i - 1].item():
            order = tran_keys[i - 1].item() < tran_keys[i].item()
            if last_order != None and order != last_order:
                # print('Need to sort')
                sorted = True
                break
            last_order = order
    # print('No need to sort')
    if sort == False:
        print('Process: Sorting keys...')
        sorted_tran_keys = torch.sort(tran_keys, 0)[0]
        print('Process: Evaluating the transformed keys...')
        evaluate_keys(sorted_tran_keys)
        if args.plot != False:
            print('Process: Plot the transofmred distribution...')
            tran_keys_plot = sample_data(tran_keys, args.train_ratio)
            plot_dist(tran_keys_plot, os.path.join(args.output_dir, output_filename + '_' + str(i) + '_tran_dist.png'))
        del sorted_tran_keys
    else:
        print('Process: Evaluating the transformed keys...')
        evaluate_keys(tran_keys)

    # print('Process: Sampling keys to build linear models...')
    # start_time = time.time()
    # num_keys = len(tran_keys)
    # max_key = torch.max(tran_keys).item()
    # min_key = torch.min(tran_keys).item()
    # y = num_keys * 1.5 * (tran_keys - min_key) / max_key
    # slope, intercept = build_linear_model(tran_keys, y)
    # intercept = -min_key * slope + 0.5
    # duration = time.time() - start_time
    # print('Time Cost of Building Linear Models', duration)
    # del y
    # conf_stat = compute_conflicts(torch.sort(tran_keys, 0)[0], slope, intercept, amp=1.5)
    # print('Total conflicts {}, max conflict {}, 99% conflict {}, average conflict {}'.format(conf_stat[0], conf_stat[1], conf_stat[2], conf_stat[3]))
    del tran_keys


if __name__ == '__main__':
    # Initializing parameters for flow models
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--input_dir', type=str, default='inputs')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--read_ratio', type=int, default=100)
    parser.add_argument('--req_dist', type=str, default='zipf')
    parser.add_argument('--dataset', type=str, default='lognormal-200M-100R-zipf')
    parser.add_argument('--seed', type=int, default=1000000007)
    parser.add_argument('--plot', type=bool, default=False)    
    parser.add_argument('--train_ratio', type=float, default=10)
    parser.add_argument('--sample_ratio', type=float, default=50)
    parser.add_argument('--num_sample', type=int, default=2)
    parser.add_argument('--num_train', type=int, default=3)

    parser.add_argument('--en_type', type=str, default=None)
    parser.add_argument('--all_pos', type=bool, default=False)
    parser.add_argument('--limited_exp', type=bool, default=False)
    parser.add_argument('--de_type', type=str, default=None)
    parser.add_argument('--trim', type=int, default=-1)

    parser.add_argument('--shifts', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--clip_norm', type=float, default=.1)
    parser.add_argument('--steps', type=int, default=15)
    parser.add_argument('--patience', type=int, default=2000)
    parser.add_argument('--decay', type=float, default=0.5)

    parser.add_argument('--paras', type=str, default='2D-2L-1H-normal')
    parser.add_argument('--loss_func', type=str, default='normal')
    parser.add_argument('--batch_dim', type=int, default=5 if print_detail else 80000)
    parser.add_argument("--flows", type=int, default=1)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=1)

    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--savefig', action='store_true')
    parser.add_argument('--reduce_extreme', action='store_true')
    args = parser.parse_args()
    args.save = True
    paras = args.paras.split('-')
    args.input_dim = int(paras[0][:-1])
    args.layers = int(paras[1][:-1]) - 1
    args.hidden_dim = int(paras[2][:-1]) // args.input_dim
    args.loss_func = paras[3]
    args.train_ratio /= 100.
    args.sample_ratio /= 100.

    args.path = os.path.join('checkpoint', '{}_{}D_{}F_{}L_{}H_{}'.format(
        args.dataset, args.input_dim, args.flows, args.layers, args.hidden_dim,
        str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')))

    random_seed = args.seed
    seed = random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    run_bnaf(args)

