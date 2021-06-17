import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import os
import math
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from utils.metrics import get_MAE, get_MSE, get_MAPE

def get_dataloader(datapath, scaler_X, scaler_Y, fraction, batch_size, mode='train'):
    datapath = os.path.join(datapath, mode)
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X = np.load(os.path.join(datapath, 'X.npy'))
    Y = np.load(os.path.join(datapath, 'Y.npy'))
    ext = np.load(os.path.join(datapath, 'ext.npy'))

    if mode == 'train' and fraction != 100:
        length = len(X)
        sample_index = int(length * fraction / 100)
        X = X[:sample_index]
        Y = Y[:sample_index]
        ext = ext[:sample_index]

    X = Tensor(np.expand_dims(X, 1)) / scaler_X
    Y = Tensor(np.expand_dims(Y, 1)) / scaler_Y
    ext = Tensor(ext)

    assert len(X) == len(Y)
    print('# {} samples: {}'.format(mode, len(X)))

    data = torch.utils.data.TensorDataset(X, ext, Y)
    if mode == 'train':
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    return dataloader

def get_dataloader_inf(datapath, scaler_X, scaler_Y, fraction, batch_size, mode='train'):
    datapath = os.path.join(datapath, mode)
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X = np.load(os.path.join(datapath, '8X.npy'))
    Y = np.load(os.path.join(datapath, 'X.npy'))
    ext = np.load(os.path.join(datapath, 'ext.npy'))

    if mode == 'train' and fraction != 100:
        length = len(X)
        sample_index = int(length * fraction / 100)
        X = X[:sample_index]
        Y = Y[:sample_index]
        ext = ext[:sample_index]

    X = Tensor(np.expand_dims(X, 1)) / scaler_X
    Y = Tensor(np.expand_dims(Y, 1)) / scaler_Y
    ext = Tensor(ext)

    assert len(X) == len(Y)
    print('# {} samples: {}'.format(mode, len(X)))

    data = torch.utils.data.TensorDataset(X, ext, Y)
    if mode == 'train':
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    return dataloader

def get_dataloader_apn(datapath, batch_size, fraction, mode='train'):
    datapath = os.path.join(datapath, mode)
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    anchor = np.load(os.path.join(datapath, 'anchor.npy'))
    pos = np.load(os.path.join(datapath, 'pos.npy'))
    neg = np.load(os.path.join(datapath, 'neg.npy'))

    if mode == 'train' and fraction != 100:
        length = len(anchor)
        sample_index = int(length * fraction / 100)
        anchor = anchor[:sample_index]
        pos = pos[:sample_index]
        neg = neg[:sample_index]

    anchor = Tensor(np.expand_dims(anchor, 1))
    pos = Tensor(np.expand_dims(pos, 1))
    neg = Tensor(np.expand_dims(neg, 1))

    assert len(anchor) == len(pos)
    print('# {} samples: {}'.format(mode, len(anchor)))

    data = torch.utils.data.TensorDataset(anchor, pos, neg)
    if mode == 'train':
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    return dataloader

def print_model_parm_nums(model, str):
    total_num = sum([param.nelement() for param in model.parameters()])
    print('{} params: {}'.format(str, total_num))

def create_tc_data_HardSample(type, mode):
    datapath = os.path.join('../data', mode)
    datapath = os.path.join(datapath, type)
    A = np.load(os.path.join(datapath, 'X.npy'))

    anchor = []
    p1 = []
    p2 = []

    length = len(A)
    for i in range(length):
        maxn = -float('inf')
        minn = float('inf')
        kmax, kmin = i, i
        for j in range(length):
            if i == j:
                continue
            dist = np.sqrt(np.sum(np.square(A[i] - A[j])))

            if dist > maxn:
                maxn = dist
                kmax = j

            if dist < minn:
                minn = dist
                kmin = j

        # print(minn, maxn, i, kmin, kmax)

        anchor.append(A[i])
        p1.append(A[kmin])
        p2.append(A[kmax])

    anchor = np.array(anchor)
    p1 = np.array(p1)
    p2 = np.array(p2)

    anchor_path = os.path.join(datapath, 'anchor.npy')
    pos_path = os.path.join(datapath, 'pos.npy')
    neg_path = os.path.join(datapath, 'neg.npy')

    np.save(anchor_path, anchor)
    np.save(pos_path, p1)
    np.save(neg_path, p2)

def create_tc_data_WeightSample(type, mode, k=5):
    datapath = os.path.join('../data', mode)
    datapath = os.path.join(datapath, type)
    A = np.load(os.path.join(datapath, 'X.npy'))

    anchor = []
    p1 = []
    p2 = []

    length = len(A)
    for i in range(length):
        dis_dict = {}

        for j in range(length):
            if i == j:
                continue
            dist = np.sqrt(np.sum(np.square(A[i] - A[j])))
            dis_dict[j] = dist

        dis_dict = sorted(dis_dict.items(), key=lambda item:item[1])
        dis_pos = dis_dict[:k]
        dis_neg = dis_dict[-k:]

        pos_sum, neg_sum = 0, 0

        for j in dis_pos:
            pos_sum += 1./j[1]

        for j in dis_neg:
            neg_sum += j[1]

        pos_zero, neg_zero = 0, 0
        for j in dis_pos:
            pos_zero += (1./j[1])/pos_sum * A[j[0]]
        for j in dis_neg:
            neg_zero += j[1]/neg_sum * A[j[0]]

        anchor.append(A[i])
        p1.append(pos_zero)
        p2.append(neg_zero)

    anchor = np.array(anchor)
    p1 = np.array(p1)
    p2 = np.array(p2)

    anchor_path = os.path.join(datapath, str(k)+'anchor.npy')
    pos_path = os.path.join(datapath, str(k)+'pos.npy')
    neg_path = os.path.join(datapath, str(k)+'neg.npy')

    np.save(anchor_path, anchor)
    np.save(pos_path, p1)
    np.save(neg_path, p2)

def create_scaler_data(type, mode):
    up_size = 4
    if mode == 'BikeNYC':
        up_size = 2

    datapath = os.path.join('../data', mode)
    datapath = os.path.join(datapath, type)
    A = np.load(os.path.join(datapath, 'X.npy'))
    (x, y, z) = A.shape
    zeros = np.zeros(shape=(x, y//up_size, z//up_size))
    for k in range(x):
        for i in range(y):
            for j in range(z):
                ii = i//up_size
                jj = j//up_size
                zeros[k][ii][jj] += A[k][i][j]

    if mode == 'BikeNYC':
        temp_path = os.path.join(datapath, '10X.npy')
        np.save(temp_path, zeros)
    else:
        temp_path = os.path.join(datapath, '8X.npy')
        np.save(temp_path, zeros)

if __name__ == '__main__':
    create_tc_data_HardSample(type='valid', mode='P1')