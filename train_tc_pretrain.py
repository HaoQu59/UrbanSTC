import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import sys
import numpy as np
import argparse
import warnings
from datetime import datetime
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from utils.metrics import get_MSE, distance_tensor
from utils.data_process import get_dataloader_apn, print_model_parm_nums
from util import weights_init_normal
from model import tc_preTrain

# load arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200,
                    help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=16,
                    help='training batch size')
parser.add_argument('--lr', type=float, default=1e-4,
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
parser.add_argument('--harved_epoch', type=int, default=20,
                    help='halved at every x interval')
parser.add_argument('--upscale_factor', type=int,
                    default=4, help='upscale factor')
parser.add_argument('--seed', type=int, default=2021, help='random seed')
parser.add_argument('--scaler_X', type=int, default=1,
                    help='scaler of coarse-grained flows')
parser.add_argument('--scaler_Y', type=int, default=1,
                    help='scaler of fine-grained flows')
parser.add_argument('--dataset', type=str, default='P1',
                    help='which dataset to use')
parser.add_argument('--margin', type=float, default=15.0)
parser.add_argument('--fraction', type=int,
                    default=100, help='fraction')

opt = parser.parse_args()
print(opt)
torch.manual_seed(opt.seed)
random.seed(opt.seed)
warnings.filterwarnings('ignore')
# path for saving model
save_path = 'Saved_model/{}/{}/{}-{}'.format(opt.dataset,
                                             opt.fraction,
                                             'SSL-tc',
                                             opt.base_channels)
os.makedirs(save_path, exist_ok=True)


# test CUDA
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# initial model
model = tc_preTrain(in_channels=opt.channels,
                    base_channels=opt.base_channels)
model.apply(weights_init_normal)
torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=5.0)
print_model_parm_nums(model, 'tc_preTrain')

if cuda:
    model.cuda()

# load training set and validation set
datapath = os.path.join('data', opt.dataset)
train_dataloader = get_dataloader_apn(
    datapath, opt.batch_size, opt.fraction, 'train')
valid_dataloader = get_dataloader_apn(
    datapath, 8, opt.fraction, 'valid')

# Optimizers
lr = opt.lr
optimizer = torch.optim.Adam(
    model.parameters(), lr=lr, betas=(opt.b1, opt.b2))

# training phase
iter = 0
InfoNce = [np.inf]
for epoch in range(opt.n_epochs):
    train_loss = 0
    ep_time = datetime.now()
    for i, (anchor, pos, neg) in enumerate(train_dataloader):
        model.train()
        optimizer.zero_grad()

        # generate images with high resolution
        anchor_tensor = model(anchor)
        pos_tensor = model(pos)
        neg_tensor = model(neg)

        d_positive = distance_tensor(anchor_tensor, pos_tensor)
        d_negative = distance_tensor(anchor_tensor, neg_tensor)

        loss = torch.clamp(opt.margin + d_positive - d_negative, min=0.0).mean()
        loss.backward()
        optimizer.step()
        print("[Epoch %d/%d] [Batch %d/%d] [Batch Loss: %f]" % (epoch,
                                                                opt.n_epochs,
                                                                i,
                                                                len(train_dataloader),
                                                                loss.item()))

        # counting training mse
        train_loss += loss.item() * len(anchor)

        iter += 1
        # validation phase
        if iter % opt.sample_interval == 0 and epoch > 0:
            model.eval()
            valid_time = datetime.now()
            total_loss = 0
            for j, (anchor, pos, neg) in enumerate(valid_dataloader):
                anchor_tensor = model(anchor)
                pos_tensor = model(pos)
                neg_tensor = model(neg)

                d_positive = distance_tensor(anchor_tensor, pos_tensor)
                d_negative = distance_tensor(anchor_tensor, neg_tensor)

                total_loss += torch.clamp(opt.margin + d_positive - d_negative, min=0.0).mean().item() * len(anchor)
            total_loss /= len(valid_dataloader.dataset)

            if total_loss < np.min(InfoNce):
                print("iter\t{}\tNCE\t{:.6f}\ttime\t{}".format(iter, total_loss, datetime.now()-valid_time))
                # save model at each iter
                # torch.save(UrbanFM.state_dict(),
                #            '{}/model-{}.pt'.format(save_path, iter))
                torch.save(model.conv_tc.state_dict(),
                           '{}/final_model.pt'.format(save_path))
                f = open('{}/results.txt'.format(save_path), 'a')
                f.write("epoch\t{}\titer\t{}\tNCE\t{:.6f}\n".format(epoch, iter, total_loss))
                f.close()
            InfoNce.append(total_loss)

    # halve the learning rate
    # if epoch % opt.harved_epoch == 0 and epoch != 0:
    #     lr /= 2
    #     optimizer = torch.optim.Adam(
    #         model.parameters(), lr=lr, betas=(opt.b1, opt.b2))
    #     f = open('{}/results.txt'.format(save_path), 'a')
    #     f.write("half the learning rate!\n")
    #     f.close()

    print('=================time cost: {}==================='.format(
        datetime.now()-ep_time))
