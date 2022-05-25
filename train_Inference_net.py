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
from utils.metrics import get_MSE
from utils.data_process import get_dataloader_inf, print_model_parm_nums
from util import weights_init_normal
from model import inference_net

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
parser.add_argument('--img_width', type=int, default=8,
                    help='image width')
parser.add_argument('--img_height', type=int, default=8,
                    help='image height')
parser.add_argument('--channels', type=int, default=1,
                    help='number of flow image channels')
parser.add_argument('--sample_interval', type=int, default=20,
                    help='interval between validation')
parser.add_argument('--harved_epoch', type=int, default=200,
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
                                             'SSL-inference',
                                             opt.base_channels)
os.makedirs(save_path, exist_ok=True)


# test CUDA
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# initial model
model = inference_net(in_channels=opt.channels,
                      out_channels=opt.channels,
                      img_width=opt.img_width,
                      img_height=opt.img_height,
                      base_channels=opt.base_channels,)
model.apply(weights_init_normal)
torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=5.0)
print_model_parm_nums(model, 'inference_net')
criterion = nn.MSELoss()

if cuda:
    model.cuda()
    criterion.cuda()

# load training set and validation set
datapath = os.path.join('data', opt.dataset)
train_dataloader = get_dataloader_inf(
    datapath, opt.scaler_X, opt.scaler_Y, opt.fraction, opt.batch_size, 'train')
valid_dataloader = get_dataloader_inf(
    datapath, opt.scaler_X, opt.scaler_Y, opt.fraction, 8, 'valid')

# Optimizers
lr = opt.lr
optimizer = torch.optim.Adam(
    model.parameters(), lr=lr, betas=(opt.b1, opt.b2))

# training phase
iter = 0
rmses = [np.inf]
maes = [np.inf]
for epoch in range(opt.n_epochs):
    train_loss = 0
    ep_time = datetime.now()
    for i, (flows_c, ext, flows_f) in enumerate(train_dataloader):
        model.train()
        optimizer.zero_grad()

        # generate images with high resolution
        gen_hr = model(flows_c)
        loss = criterion(gen_hr, flows_f)

        loss.backward()
        optimizer.step()

        print("[Epoch %d/%d] [Batch %d/%d] [Batch Loss: %f]" % (epoch,
                                                                opt.n_epochs,
                                                                i,
                                                                len(train_dataloader),
                                                                np.sqrt(loss.item())))

        # counting training mse
        train_loss += loss.item() * len(flows_c)

        iter += 1
        # validation phase
        if iter % opt.sample_interval == 0:
            model.eval()
            valid_time = datetime.now()
            total_mse, total_mae = 0, 0
            for j, (flows_c, ext, flows_f) in enumerate(valid_dataloader):
                preds = model(flows_c)
                preds = preds.cpu().detach().numpy()
                flows_f = flows_f.cpu().detach().numpy()
                total_mse += get_MSE(preds, flows_f) * len(flows_c)
            rmse = np.sqrt(total_mse / len(valid_dataloader.dataset))
            if rmse < np.min(rmses):
                print("iter\t{}\tRMSE\t{:.6f}\ttime\t{}".format(iter, rmse, datetime.now()-valid_time))
                # save model at each iter
                # torch.save(UrbanFM.state_dict(),
                #            '{}/model-{}.pt'.format(save_path, iter))
                torch.save(model.conv1.state_dict(),
                           '{}/final_model.pt'.format(save_path))
                f = open('{}/results.txt'.format(save_path), 'a')
                f.write("epoch\t{}\titer\t{}\tRMSE\t{:.6f}\n".format(epoch, iter, rmse))
                f.close()
            rmses.append(rmse)

    print('=================time cost: {}==================='.format(
        datetime.now()-ep_time))
