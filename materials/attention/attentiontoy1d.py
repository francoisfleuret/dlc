#!/usr/bin/env python
# coding=utf-8

# @XREMOTE_HOST: elk.fleuret.org
# @XREMOTE_EXEC: /home/fleuret/conda/bin/python
# @XREMOTE_PRE: killall -q -9 python || echo "Nothing killed"
# @XREMOTE_GET: *.pdf *.log

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

import os
import sys
import math
import random
import argparse

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

COLORS = mpl.rcParams['axes.prop_cycle'].by_key()['color']
COLOR_INPUT = COLORS[1] # blue
COLOR_TARGET = COLORS[0] # red
COLOR_OUTPUT = COLORS[3] #

nb_seq_to_save = 15

######################################################################

parser = argparse.ArgumentParser(description = 'Toy attention model.')

parser.add_argument('--nb_epochs',
                    type = int, default = 250)

parser.add_argument('--with_attention',
                    help = 'Use the model with an attention layer',
                    action = 'store_true', default = False)

parser.add_argument('--group_by_locations',
                    help = 'Use the task where the grouping is location-based',
                    action = 'store_true', default = False)

parser.add_argument('--positional_encoding',
                    help = 'Provide a positional encoding',
                    action = 'store_true', default = False)

parser.add_argument('--single',
                    action = 'store_true',
                    help = 'When false, run all experiments')

parser.add_argument('--seed',
                    type = int, default = 1, # In original code, seed = 1
                    help = 'Random seed, < 0 is no seeding')

parser.add_argument('--compute',
                    action = 'store_true',
                    help = 'Whether to compute')

parser.add_argument('--draw',
                    default = '',
                    help = 'Where to save images')

parser.add_argument('--results_dir',
                    default = 'data',
                    help = 'Directory where to save results')

parser.add_argument('--image_dir',
                    default = os.path.join('pics', 'imgs'),
                    help = 'Directory of persistent images')

parser.add_argument('--ext',
                    default = 'pdf',
                    help = 'Output extension (default: pdf)')

args = parser.parse_args()

if args.compute: sys.exit(0) # --draw does the computation to avoid too much modification
if args.draw: os.makedirs(args.draw, exist_ok = True)
if args.results_dir: os.makedirs(args.results_dir, exist_ok = True)

######################################################################

def plot_loss(figname, losses, ymax = 1700):
    '''Plot input loss curves.

    losses = list of (datname, color, label)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(1, args.nb_epochs)
    ax.set_xscale('log')
    ax.set_ylim(0, ymax)
    ax.set_xlabel('Nb. of epochs')
    ax.set_ylabel('MSE')

    for datname, color, label in losses:
        data = np.loadtxt(datname)
        ax.plot(data[:,0], data[:,1], label = label, c = color)

    ax.grid()
    ax.legend()

    print(f'Saving {figname}')
    fig.savefig(figname, bbox_inches = 'tight')
    plt.close(fig)


# If we don't specify '--single' all experiments of the original
# run-attentiontoy1d.sh are run.
if not args.single:
    os.system(f'{" ".join(sys.argv)} --single ')
    os.system(f'{" ".join(sys.argv)} --single --with_attention')

    plot_loss(os.path.join(args.draw, f'att1d_train_log.{args.ext}'),
              [
                  (os.path.join(args.results_dir, f'att1d_train.dat'), COLORS[0], 'Without attention'),
              ],
              1700
    )
    plot_loss(os.path.join(args.draw, f'att1d_wa_train_log.{args.ext}'),
              [
                  (os.path.join(args.results_dir, f'att1d_train.dat'), COLORS[0], 'Without attention'),
                  (os.path.join(args.results_dir, f'att1d_wa_train.dat'), COLORS[1], 'With attention'),
              ],
              1700
    )

    os.system(f'{" ".join(sys.argv)} --single --with_attention --group_by_locations')
    os.system(f'{" ".join(sys.argv)} --single --with_attention --group_by_locations --positional_encoding')


    plot_loss(os.path.join(args.draw, f'att1d_wa_lg_train_log.{args.ext}'),
              [
                  (os.path.join(args.results_dir, f'att1d_wa_lg_train.dat'), COLORS[2], 'With attention, no positional encoding'),
              ],
              2700
    )
    plot_loss(os.path.join(args.draw, f'att1d_wa_lg_pe_train_log.{args.ext}'),
              [
                  (os.path.join(args.results_dir, f'att1d_wa_lg_train.dat'), COLORS[2], 'With attention, no positional encoding'),
                  (os.path.join(args.results_dir, f'att1d_wa_lg_pe_train.dat'), COLORS[3], 'With attention, positional encoding'),
              ],
              2700
    )
    sys.exit(0)

######################################################################

label = ''

if args.with_attention: label = 'wa_'

if args.group_by_locations: label += 'lg_'

if args.positional_encoding: label += 'pe_'

log_filename = os.path.join(args.results_dir, f'att1d_{label}train.dat')
log_file = open(log_filename, 'w')

######################################################################

def log_string(s):
    if log_file is not None:
        log_file.write(s + '\n')
        log_file.flush()
    print(s)
    sys.stdout.flush()

def set_seed(seed):
    if seed >= 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

######################################################################

if torch.cuda.is_available():
    device = torch.device('cuda')
    # torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

set_seed(args.seed)

######################################################################

seq_height_min, seq_height_max = 1.0, 25.0
seq_width_min, seq_width_max = 5.0, 11.0
seq_length = 100

def positions_to_sequences(tr = None, bx = None, noise_level = 0.3):
    st = torch.arange(seq_length, device = device).float()
    st = st[None, :, None]
    tr = tr[:, None, :, :]
    bx = bx[:, None, :, :]

    xtr =            torch.relu(tr[..., 1] - torch.relu(torch.abs(st - tr[..., 0]) - 0.5) * 2 * tr[..., 1] / tr[..., 2])
    xbx = torch.sign(torch.relu(bx[..., 1] - torch.abs((st - bx[..., 0]) * 2 * bx[..., 1] / bx[..., 2]))) * bx[..., 1]

    x = torch.cat((xtr, xbx), 2)

    # u = x.sign()
    u = F.max_pool1d(x.sign().permute(0, 2, 1), kernel_size = 2, stride = 1).permute(0, 2, 1)

    collisions = (u.sum(2) > 1).max(1).values
    y = x.max(2).values

    return y + torch.rand_like(y) * noise_level - noise_level / 2, collisions

######################################################################

def generate_sequences(nb):

    # Position / height / width

    tr = torch.empty(nb, 2, 3, device = device)
    tr[:, :, 0].uniform_(seq_width_max/2, seq_length - seq_width_max/2)
    tr[:, :, 1].uniform_(seq_height_min, seq_height_max)
    tr[:, :, 2].uniform_(seq_width_min, seq_width_max)

    bx = torch.empty(nb, 2, 3, device = device)
    bx[:, :, 0].uniform_(seq_width_max/2, seq_length - seq_width_max/2)
    bx[:, :, 1].uniform_(seq_height_min, seq_height_max)
    bx[:, :, 2].uniform_(seq_width_min, seq_width_max)

    if args.group_by_locations:
        a = torch.cat((tr, bx), 1)
        v = a[:, :, 0].sort(1).values[:, 2:3]
        mask_left = (a[:, :, 0] < v).float()
        h_left = (a[:, :, 1] * mask_left).sum(1) / 2
        h_right = (a[:, :, 1] * (1 - mask_left)).sum(1) / 2
        valid = (h_left - h_right).abs() > 4
    else:
        valid = (torch.abs(tr[:, 0, 1] - tr[:, 1, 1]) > 4) & (torch.abs(tr[:, 0, 1] - tr[:, 1, 1]) > 4)

    input, collisions = positions_to_sequences(tr, bx)

    if args.group_by_locations:
        a = torch.cat((tr, bx), 1)
        v = a[:, :, 0].sort(1).values[:, 2:3]
        mask_left = (a[:, :, 0] < v).float()
        h_left = (a[:, :, 1] * mask_left).sum(1, keepdim = True) / 2
        h_right = (a[:, :, 1] * (1 - mask_left)).sum(1, keepdim = True) / 2
        a[:, :, 1] = mask_left * h_left + (1 - mask_left) * h_right
        tr, bx = a.split(2, 1)
    else:
        tr[:, :, 1:2] = tr[:, :, 1:2].mean(1, keepdim = True)
        bx[:, :, 1:2] = bx[:, :, 1:2].mean(1, keepdim = True)

    targets, _ = positions_to_sequences(tr, bx)

    valid = valid & ~collisions
    tr = tr[valid]
    bx = bx[valid]
    input = input[valid][:, None, :]
    targets = targets[valid][:, None, :]

    if input.size(0) < nb:
        input2, targets2, tr2, bx2 = generate_sequences(nb - input.size(0))
        input = torch.cat((input, input2), 0)
        targets = torch.cat((targets, targets2), 0)
        tr = torch.cat((tr, tr2), 0)
        bx = torch.cat((bx, bx2), 0)

    return input, targets, tr, bx

######################################################################

def save_sequence_images(filename, sequences, tr = None, bx = None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlim(0, seq_length + 5) # Olivier: for pretty grid
    ax.set_ylim(-1, seq_height_max + 4)

    for u in sequences:
        ax.plot(
            torch.arange(u[0].size(0)) + 0.5, u[0], color = u[1], label = u[2]
        )

    ax.legend(frameon = False, loc = 'upper left')
    ax.grid()

    delta = -1.
    if tr is not None:
        ax.scatter(tr[:, 0].cpu(), torch.full((tr.size(0),), delta), color = 'black', marker = '^', clip_on = False)

    if bx is not None:
        ax.scatter(bx[:, 0].cpu(), torch.full((bx.size(0),), delta), color = 'black', marker = 's', clip_on = False)

    fig.savefig(filename, bbox_inches = 'tight')

    plt.close('all')

######################################################################

def dummy():
# START_EINSTEIN_ATTENTION
    A = torch.einsum('nct,ncs->nts', Q, K).softmax(2)
    y = torch.einsum('nts,ncs->nct', A, V)
# END_EINSTEIN_ATTENTION

# START_ATTENTION_LAYER
class SelfAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, key_dim):
        super().__init__()
        self.conv_Q = nn.Conv1d(in_dim, key_dim, kernel_size = 1, bias = False)
        self.conv_K = nn.Conv1d(in_dim, key_dim, kernel_size = 1, bias = False)
        self.conv_V = nn.Conv1d(in_dim, out_dim, kernel_size = 1, bias = False)

    def forward(self, x):
        Q = self.conv_Q(x)
        K = self.conv_K(x)
        V = self.conv_V(x)
        A = Q.transpose(1, 2).matmul(K).softmax(2)
        y = A.matmul(V.transpose(1, 2)).transpose(1, 2)
        return y
# END_ATTENTION_LAYER

    def __repr__(self):
        return self._get_name() + \
            '(in_dim={}, out_dim={}, key_dim={})'.format(
                self.conv_Q.in_dim,
                self.conv_V.out_dim,
                self.conv_K.out_dim
            )

    def attention(self, x):
        Q = self.conv_Q(x)
        K = self.conv_K(x)
        return Q.transpose(1, 2).matmul(K).softmax(2)

######################################################################

train_input, train_targets, train_tr, train_bx = generate_sequences(25000)
test_input, test_targets, test_tr, test_bx = generate_sequences(1000)

######################################################################

ks = 5
nc = 64

if args.positional_encoding:
    c = math.ceil(math.log(seq_length) / math.log(2.0))
    o = 2**torch.arange(c).unsqueeze(1)
    positional_input = (torch.arange(seq_length).unsqueeze(0).div(o, rounding_mode = 'floor')) % 2
    positional_input = positional_input.unsqueeze(0).float()
else:
    positional_input = torch.zeros(1, 0, seq_length)

in_dim = 1 + positional_input.size(1)

if args.with_attention:

    model = nn.Sequential(
        nn.Conv1d(in_dim, nc, kernel_size = ks, padding = ks//2),
        nn.ReLU(),
        nn.Conv1d(nc, nc, kernel_size = ks, padding = ks//2),
        nn.ReLU(),
        SelfAttentionLayer(nc, nc, nc),
        nn.Conv1d(nc, nc, kernel_size = ks, padding = ks//2),
        nn.ReLU(),
        nn.Conv1d(nc,  1, kernel_size = ks, padding = ks//2)
    )

else:

    model = nn.Sequential(
        nn.Conv1d(in_dim, nc, kernel_size = ks, padding = ks//2),
        nn.ReLU(),
        nn.Conv1d(nc, nc, kernel_size = ks, padding = ks//2),
        nn.ReLU(),
        nn.Conv1d(nc, nc, kernel_size = ks, padding = ks//2),
        nn.ReLU(),
        nn.Conv1d(nc, nc, kernel_size = ks, padding = ks//2),
        nn.ReLU(),
        nn.Conv1d(nc,  1, kernel_size = ks, padding = ks//2)
    )

nb_parameters = sum(p.numel() for p in model.parameters())

with open(f'att1d_{label}model.log', 'w') as f:
    f.write(str(model) + '\n\n')
    f.write(f'nb_parameters {nb_parameters}\n')

######################################################################

# START_TRAINING
batch_size = 100

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
mse_loss = nn.MSELoss()

model.to(device)                                                              # HIDE_IN_SLIDE
mse_loss.to(device)                                                           # HIDE_IN_SLIDE
train_input, train_targets = train_input.to(device), train_targets.to(device) # HIDE_IN_SLIDE
test_input, test_targets = test_input.to(device), test_targets.to(device)     # HIDE_IN_SLIDE
positional_input = positional_input.to(device)                                # HIDE_IN_SLIDE
                                                                              # HIDE_IN_SLIDE

mu, std = train_input.mean(), train_input.std()

for e in range(args.nb_epochs):
    acc_loss = 0.0 # HIDE_IN_SLIDE

    for input, targets in zip(train_input.split(batch_size),
                              train_targets.split(batch_size)):

        input = torch.cat((input, positional_input.expand(input.size(0), -1, -1)), 1) # HIDE_IN_SLIDE
                                                                                      # HIDE_IN_SLIDE

        output = model((input - mu) / std)
        loss = mse_loss(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
# END_TRAINING

        acc_loss += loss.item()

    log_string(f'{e+1} {acc_loss}')

######################################################################

train_input = train_input.detach().to('cpu')
train_targets = train_targets.detach().to('cpu')

for k in range(15):
    save_sequence_images(
        os.path.join(args.draw, f'att1d_{label}train_{k:03d}.{args.ext}'),
        [
            ( train_input[k, 0], COLOR_INPUT, 'Input' ),
            ( train_targets[k, 0], COLOR_TARGET, 'Target' ),
        ],
    )

####################

test_input = torch.cat((test_input, positional_input.expand(test_input.size(0), -1, -1)), 1)
test_outputs = model((test_input - mu) / std).detach()

if args.with_attention:
    k = next(k for k, l in enumerate(model) if isinstance(l, SelfAttentionLayer))
    x = model[0:k]((test_input - mu) / std)
    test_A = model[k].attention(x)
    test_A = test_A.detach().to('cpu')

test_input = test_input.detach().to('cpu')
test_outputs = test_outputs.detach().to('cpu')
test_targets = test_targets.detach().to('cpu')

for k in range(15):
    save_sequence_images(
        os.path.join(args.draw, f'att1d_{label}test_Y_{k:03d}.{args.ext}'),
        [
            ( test_input[k, 0], COLOR_INPUT, 'Input' ),
            ( test_outputs[k, 0], COLOR_OUTPUT, 'Output' ),
        ]
    )

    save_sequence_images(
        os.path.join(args.draw, f'att1d_{label}test_Yp_{k:03d}.{args.ext}'),
        [
            ( test_input[k, 0], COLOR_INPUT, 'Input' ),
            ( test_outputs[k, 0], COLOR_OUTPUT, 'Output' ),
        ],
        test_tr[k],
        test_bx[k]
    )

    if args.with_attention:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(0, seq_length)
        ax.set_ylim(0, seq_length)

        for loc in [ 'left', 'right', 'top', 'bottom' ]:
            ax.spines[loc].set_visible(True)

        ax.imshow(test_A[k], cmap = 'binary', interpolation = 'nearest')
        delta = 0.
        ax.scatter(test_bx[k, :, 0].cpu(), torch.full((test_bx.size(1),), delta), color = 'black', marker = 's', clip_on = False)
        ax.scatter(torch.full((test_bx.size(1),), delta), test_bx[k, :, 0].cpu(), color = 'black', marker = 's', clip_on = False)
        ax.scatter(test_tr[k, :, 0].cpu(), torch.full((test_tr.size(1),), delta), color = 'black', marker = '^', clip_on = False)
        ax.scatter(torch.full((test_tr.size(1),), delta), test_tr[k, :, 0].cpu(), color = 'black', marker = '^', clip_on = False)

        figname = os.path.join(args.draw, f'att1d_{label}test_A_{k:03d}.{args.ext}')
        fig.savefig(figname, bbox_inches = 'tight')

    plt.close('all')

######################################################################
