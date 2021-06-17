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

from utils.metrics import loss_c
from utils.data_process import get_dataloader, print_model_parm_nums
from util import weights_init_normal
from model import reg_preTrain

# load arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=100,
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
parser.add_argument('--margin', type=float, default=1e-4, help='1e-4 or 5e-5')
parser.add_argument('--type', type=str, default='softmax', help='sigmoid or softmax')
parser.add_argument('--temperature', type=float, default=0.1,
                    help='Temperature parameter for InfoNCE.')
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
                                             'SSL-reg',
                                             opt.base_channels)
os.makedirs(save_path, exist_ok=True)

# test CUDA
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# initial model
model = reg_preTrain(in_channels=opt.channels,
                     base_channels=opt.base_channels)
model.apply(weights_init_normal)
torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=5.0)
print_model_parm_nums(model, 'reg_preTrain')

if cuda:
    model.cuda()

# load training set and validation set
datapath = os.path.join('data', opt.dataset)
train_dataloader = get_dataloader(
    datapath, opt.scaler_X, opt.scaler_Y, opt.fraction, opt.batch_size, 'train')
valid_dataloader = get_dataloader(
    datapath, opt.scaler_X, opt.scaler_Y, opt.fraction, 8, 'valid')

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
    for i, (flows_c, ext, flows_f) in enumerate(train_dataloader):
        model.train()
        optimizer.zero_grad()

        # generate images with high resolution
        c_tensor = model(flows_c)
        loss = loss_c(c_tensor, margin=opt.margin, Type=opt.type)
        loss.backward()
        optimizer.step()
        print("[Epoch %d/%d] [Batch %d/%d] [Batch Loss: %f]" % (epoch,
                                                                opt.n_epochs,
                                                                i,
                                                                len(train_dataloader),
                                                                loss.item()))

        # counting training mse
        train_loss += loss.item() * len(flows_c)

        iter += 1
        # validation phase
        if iter % opt.sample_interval == 0 and epoch > 0:
            model.eval()
            valid_time = datetime.now()
            total_loss = 0
            for j, (flows_c, ext, flows_f) in enumerate(valid_dataloader):
                c_tensor = model(flows_c)
                total_loss += loss_c(c_tensor, margin=opt.margin, Type=opt.type).item() * len(flows_c)
            total_loss /= len(valid_dataloader.dataset)

            if total_loss < np.min(InfoNce):
                print("iter\t{}\tNCE\t{:.6f}\ttime\t{}".format(iter, total_loss, datetime.now() - valid_time))
                # save model at each iter
                # torch.save(UrbanFM.state_dict(),
                #            '{}/model-{}.pt'.format(save_path, iter))
                torch.save(model.conv_pix.state_dict(),
                           '{}/final_model.pt'.format(save_path))
                f = open('{}/results.txt'.format(save_path), 'a')
                f.write("epoch\t{}\titer\t{}\tNCE\t{:.6f}\n".format(epoch, iter, total_loss))
                f.close()
            InfoNce.append(total_loss)

    # halve the learning rate
    if epoch % opt.harved_epoch == 0 and epoch != 0:
        lr /= 2
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=(opt.b1, opt.b2))
        f = open('{}/results.txt'.format(save_path), 'a')
        f.write("half the learning rate!\n")
        f.close()

    print('=================time cost: {}==================='.format(
        datetime.now()-ep_time))
