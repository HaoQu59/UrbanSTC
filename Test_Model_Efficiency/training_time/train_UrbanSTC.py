import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import sys
import warnings
import numpy as np
import argparse
from datetime import datetime
from PIL import Image

import torch
import torch.nn as nn

from utils.metrics import get_MSE, get_MAE, get_MAPE
from utils.data_process import get_dataloader, print_model_parm_nums
from util import weights_init_normal
from model import UrbanSTC
from tensorboardX import SummaryWriter

# load arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=30,
                    help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=16,
                    help='training batch size')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.9,
                    help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999,
                    help='adam: decay of second order momentum of gradient')
parser.add_argument('--base_channels', type=int,
                    default=128, help='number of feature maps')
parser.add_argument('--img_width', type=int, default=32,
                    help='image width')
parser.add_argument('--img_height', type=int, default=32,
                    help='image height')
parser.add_argument('--channels', type=int, default=1,
                    help='number of flow image channels')
parser.add_argument('--sample_interval', type=int, default=20,
                    help='interval between validation')
parser.add_argument('--harved_epoch', type=int, default=50,
                    help='halved at every x interval')
parser.add_argument('--upscale_factor', type=int,
                    default=4, help='upscale factor')
parser.add_argument('--seed', type=int, default=2021, help='random seed')
parser.add_argument('--extend', type=int, default=1,
                    help='combine tensor dim', choices='1,2,3')
parser.add_argument('--scaler_X', type=int, default=1,
                    help='scaler of coarse-grained flows')
parser.add_argument('--scaler_Y', type=int, default=1,
                    help='scaler of fine-grained flows')
parser.add_argument('--ext_flag', type=bool, default=True,
                    help='external factors')
parser.add_argument('--dataset', type=str, default='P1',
                    help='which dataset to use')
parser.add_argument('--change_epoch', type=int, default=5,
                    help='change optimizer')
parser.add_argument('--fraction', type=int,
                    default=100, help='fraction')


opt = parser.parse_args()
print(opt)
torch.manual_seed(opt.seed)
warnings.filterwarnings('ignore')

# test CUDA
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# initial model
model = UrbanSTC(in_channels=opt.channels,
                 out_channels=opt.channels,
                 img_width=opt.img_width,
                 img_height=opt.img_height,
                 base_channels=opt.base_channels,
                 extend=opt.extend,
                 ext_flag=opt.ext_flag)
model.apply(weights_init_normal)
torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=5.0)

print_model_parm_nums(model, 'UrbanSTC')

criterion = nn.MSELoss()

if cuda:
    model.cuda()
    criterion.cuda()

# load training set and validation set
datapath = os.path.join('../../data', opt.dataset)
train_dataloader = get_dataloader(
    datapath, opt.scaler_X, opt.scaler_Y, opt.fraction, opt.batch_size, 'train')
valid_dataloader = get_dataloader(
    datapath, opt.scaler_X, opt.scaler_Y, opt.fraction, 8, 'valid')

# Optimizers
for p in model.conv1.parameters():
    p.requires_grad = False
for p in model.conv_tc.parameters():
    p.requires_grad = False
for p in model.conv_pix.parameters():
    p.requires_grad = False

lr = opt.lr
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(opt.b1, opt.b2))

# training phase
iter = 0
rmses = [np.inf]
maes = [np.inf]
ep_time = datetime.now()
for epoch in range(opt.n_epochs):
    train_loss = 0

    if epoch == opt.change_epoch:
        for p in model.conv1.parameters():
            p.requires_grad = True
        for p in model.conv_tc.parameters():
            p.requires_grad = True
        for p in model.conv_pix.parameters():
            p.requires_grad = True

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=(opt.b1, opt.b2))

    for i, (flows_c, ext, flows_f) in enumerate(train_dataloader):
        model.train()
        optimizer.zero_grad()

        # generate images with high resolution
        gen_hr = model(flows_c, ext)
        loss = criterion(gen_hr, flows_f)

        loss.backward()
        optimizer.step()

        # counting training mse
        train_loss += loss.item() * len(flows_c)

        iter += 1

    # halve the learning rate
    if epoch % opt.harved_epoch == 0 and epoch != 0:
        lr /= 2
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=(opt.b1, opt.b2))

    print('=================Epoch:{} Average time cost: {}==================='.format(
        epoch + 1, (datetime.now() - ep_time) / (epoch + 1)))