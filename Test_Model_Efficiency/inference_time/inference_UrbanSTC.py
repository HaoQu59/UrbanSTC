import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import warnings
import argparse
import numpy as np
from datetime import datetime

import torch
from utils.metrics import get_MAE, get_MSE, get_MAPE
from utils.data_process import get_dataloader, print_model_parm_nums

from model import *

warnings.filterwarnings("ignore")
# load arguments
parser = argparse.ArgumentParser()
parser.add_argument('--base_channels', type=int,
                    default=128, help='number of feature maps')
parser.add_argument('--img_width', type=int, default=32,
                    help='image width')
parser.add_argument('--img_height', type=int, default=32,
                    help='image height')
parser.add_argument('--channels', type=int, default=1,
                    help='number of flow image channels')
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
parser.add_argument('--fraction', type=int,
                    default=100, help='fraction')
opt = parser.parse_args()
print(opt)

model_path = '../../Saved_model/{}/{}/{}-{}-{}'.format(opt.dataset,
                                                       opt.fraction,
                                                       'UrbanSTC',
                                                       opt.base_channels,
                                                       opt.ext_flag)

# test CUDA
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# load mode
model = UrbanSTC(in_channels=opt.channels,
                 out_channels=opt.channels,
                 img_width=opt.img_width,
                 img_height=opt.img_height,
                 base_channels=opt.base_channels,
                 extend=opt.extend,
                 ext_flag=opt.ext_flag)

model.load_state_dict(torch.load('{}/final_model.pt'.format(model_path)))
model.eval()
if cuda:
    model.cuda()
print_model_parm_nums(model, 'UrbanSTC')

# load test set
datapath = os.path.join('../../data', opt.dataset)
dataloader = get_dataloader(
    datapath, opt.scaler_X, opt.scaler_Y, fraction=100, batch_size=16, mode='test')

ep_time = datetime.now()

for j, (test_data, ext, test_labels) in enumerate(dataloader):
    preds = model(test_data, ext).cpu().detach().numpy() * opt.scaler_Y
    test_labels = test_labels.cpu().detach().numpy() * opt.scaler_Y

print('=================Inference time cost: {}==================='.format(
        datetime.now()-ep_time))