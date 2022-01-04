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

def create_model(args, verbose=False):
    
    flows = []
    for f in range(args.flows):
        layers = [MaskedWeight(args.input_dim, args.input_dim * args.hidden_dim, dim=args.input_dim, bias=False), Tanh()]
        for _ in range(args.layers - 1):
            layers.append(MaskedWeight(args.input_dim * args.hidden_dim, args.input_dim * args.hidden_dim, dim=args.input_dim, bias=False))
            layers.append(Tanh())
        layers += [MaskedWeight(args.input_dim * args.hidden_dim, args.input_dim, dim=args.input_dim, bias=False)]

        flows.append(
            BNAF(*(layers), res='gated' if f < args.flows - 1 else False)
        )

        if f < args.flows - 1:
            flows.append(Permutation(args.input_dim, 'flip'))

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
    log_p_y_mb = torch.distributions.Normal(mean, var).log_prob(y_mb).sum(-1)
    return log_p_y_mb + log_diag_j_mb


def train_density1d(data_loader, model, optimizer, scheduler, args):
    best_loss = 1e20
    model.train()

    for epoch in range(args.steps):
        print('Epoch:{}'.format(epoch))
        epoch_loss = 0
        for i, x in enumerate(data_loader):
            x_mb = x[0].to(args.device)
            loss = -compute_log_p_x(model, x_mb, args).mean()
            if np.isnan(loss.item()):
                print('Loss is NaN')
                return False
            epoch_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)

            optimizer.step()
            optimizer.zero_grad()
            
            scheduler.step(loss)

        print('Loss:{}'.format(epoch_loss / len(data_loader)))
        if np.isinf(epoch_loss):
            print('Loss is INF')
            return False

        if np.fabs(best_loss - epoch_loss) < 0.01 * np.fabs(best_loss):
            print('Early stop at epoch-{}'.format(epoch))
            break
        if epoch_loss < best_loss:
            print('Saving model at epoch-{}'.format(epoch))
            best_loss = epoch_loss
    return True


def test_density1d(data_loader, model, args):
    start = time.time()
    model.eval()

    res = []
    with torch.no_grad():
        for i, x in enumerate(data_loader):
            x_mb = x[0].to(args.device)
            y_mb = model(x_mb)[:, -1]
            if torch.isnan(y_mb).any():
                print('NAN KEYS')
                exit()
            res += y_mb.squeeze().cpu().numpy().tolist()
        print('Evaluation time {} ns'.format((time.time() - start) * 1e9 / len(res)))
        return torch.Tensor(res)


def load(model, optimizer, path):
    print('Process: Loading model...')
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def save(model, optimizer, path):
    print('Process: Saving model [' + path + ']...')
    d = {}
    d["model"] = model.state_dict()
    d["optimizer"] = optimizer.state_dict()
    torch.save(d, path)


def save_weights(dataset, res_dir, output_filename, dims, model, mean, var):
    weight_output_file = os.path.join(res_dir, output_filename + '-weights.txt')
    f = open(weight_output_file, 'w')
    f.write('%d\t%.16f\t%.16f\n' % (1, mean.item(), var.item()))
    
    model.eval()
    f.write('%d\t%d\t%d\n' % (dims[0], dims[1], dims[2]))
    model.save_weights(f)
    f.close()


def transform_workload(keys, args, model, save_path, sort=False):
    if args.input_dim == 2:
        keys = seg_data(keys, True)
    batch_dim = int(np.minimum(keys.shape[0], args.batch_dim))
    data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(keys), batch_size=batch_dim, num_workers=16)
    tran_keys = test_density1d(data_loader, model, args).squeeze().unsqueeze(1)

    if sort:
        tran_keys = torch.sort(tran_keys, 0)[0]
        evaluate_keys(tran_keys)
        
    if args.save_keys:
        print('Process: Saving the transformed keys...')
        np.savetxt(os.path.join(args.output_dir, save_path), tran_keys.squeeze().cpu().numpy(), fmt='%.32f')
    del tran_keys


def sample(args, model):
    mean = torch.zeros(args.input_dim).to(args.device)
    var = torch.ones(args.input_dim).to(args.device)
    sample_keys = torch.distributions.Normal(mean, var).sample(sample_shape=(args.sample_number, )).to(args.device)
    # Normalizing to [-1, 1]
    sample_keys = 2 * (sample_keys - torch.min(sample_keys)) / (torch.max(sample_keys) - torch.min(sample_keys)) - 1
    model.eval()
    return model.inverse(sample_keys)


def run_bnaf(args):
    # Set output file
    # output_filename = args.dataset + '-' + ('NS-' if args.seg else '1S-') + str(args.input_dim) + 'D' + '-' + str(args.layers) + 'L-' + str(args.hidden_dim * args.input_dim) + 'H-' + args.loss_func
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
    train_keys = load_data(os.path.join(args.input_dir, args.dataset))
    train_keys, _ = train_data(train_keys, args.input_dim, 0.1, 'point', dim=args.batch_dim)

    print('Process: Normalizing data...')
    if args.normalize != None:
        mean = torch.min(load_keys)
        var = (torch.max(load_keys) - torch.min(load_keys)) / args.normalize
        print('Mean', mean)
        print('Variance', var)
        load_keys = (load_keys - mean) / var

    print('Process: Creating BNAF model...')
    model = create_model(args, verbose=True)
    print('Process: Creating optimizer...')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    dims = [args.layers, args.input_dim, args.input_dim * args.hidden_dim]

    if args.load:
        print('Process: Loading model...')
        load(model, optimizer, os.path.join(args.load, 'checkpoint.pt'))
    else:
        print('Process: Creating scheduler...')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.decay,
                                                                patience=args.patience,
                                                                min_lr=5e-4, verbose=True,
                                                                threshold_mode='abs')
        batch_dim = int(np.minimum(train_keys.shape[0], args.batch_dim))
        data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_keys), batch_size=batch_dim, num_workers=16)
        print('Process: Training...')
        print('Number of training keys {}'.format(train_keys.shape[0]))
        success = train_density1d(data_loader, model, optimizer, scheduler, args)
        if not success:
            return 
        save(model, optimizer, os.path.join(args.load or args.path, 'checkpoint.pt'))
    
    gen_keys = sample(args, model)


if __name__ == '__main__':
    # Initializing parameters for flow models
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--input_dir', type=str, default='inputs')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--read_ratio', type=int, default=100)
    parser.add_argument('--req_dist', type=str, default='zipf')
    parser.add_argument('--dataset', type=str, default='lognormal-19M-var(0.5)-100R-zipf')
    parser.add_argument('--seed', type=int, default=1000000007)
    parser.add_argument('--save_keys', type=bool, default=False)
    parser.add_argument('--sample_number', type=int, default=100000)

    parser.add_argument('--normalize', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--clip_norm', type=float, default=.1)
    parser.add_argument('--steps', type=int, default=15)
    parser.add_argument('--patience', type=int, default=2000)
    parser.add_argument('--decay', type=float, default=0.5)

    parser.add_argument('--paras', type=str, default='1S-2D-2L-1H-normal')
    parser.add_argument('--lbd1', type=float, default=5)
    parser.add_argument('--lbd2', type=float, default=5)
    parser.add_argument('--seg', type=bool, default=False)
    parser.add_argument('--loss_func', type=str, default='normal')
    parser.add_argument('--batch_dim', type=int, default=2048)
    parser.add_argument('--block_dim', type=int, default=32)
    parser.add_argument('--flows', type=int, default=1)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=1)

    parser.add_argument('--load', type=str, default=None) # './checkpoint/lognormal-19M-var(0.5)-100R-zipf_2D_2L_1H_2021-10-10-21-24-42')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--savefig', action='store_true')
    parser.add_argument('--reduce_extreme', action='store_true')
    args = parser.parse_args()
    args.save = True
    paras = args.paras.split('-')
    args.seg = (paras[0][0] != '1')
    args.input_dim = int(paras[1][:-1])
    args.layers = int(paras[2][:-1])
    args.hidden_dim = int(paras[3][:-1])
    args.loss_func = paras[4]

    args.path = os.path.join('checkpoint', '{}_{}D_{}L_{}H_{}'.format(
        args.dataset, args.input_dim, args.layers, args.hidden_dim,
        str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')))

    random_seed = args.seed
    seed = random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    run_bnaf(args)

