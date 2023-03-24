### The following code has been adapated from:
### Ilia Zaitsev. Deep time series classification. https://www.kaggle.com/purplejester/pytorch-deep-time-series-classification, 2019.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import TensorDataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as func
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader


class _SepConv1d(nn.Module):
    """A simple separable convolution implementation.
    
    The separable convlution is a method to reduce number of the parameters 
    in the deep learning network for slight decrease in predictions quality.
    """
    def __init__(self, ni, no, kernel, stride, pad):
        super().__init__()
        self.depthwise = nn.Conv1d(ni, ni, kernel, stride, padding=pad, groups=ni)
        self.pointwise = nn.Conv1d(ni, no, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
    
    
    
class SepConv1d(nn.Module):
    """Implementes a 1-d convolution with 'batteries included'.
    
    The module adds (optionally) activation function and dropout layers right after
    a separable convolution layer.
    """
    def __init__(self, ni, no, kernel, stride, pad, drop=None,
                 activ=lambda: nn.ReLU(inplace=True)):
    
        super().__init__()
       # assert drop is None or (0.0 < drop < 1.0)
        layers = [_SepConv1d(ni, no, kernel, stride, pad)]
        if activ:
            layers.append(activ())
        if drop is not None:
            layers.append(nn.Dropout(drop))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x): 
       # print('bp_conv1')
        return self.layers(x)
    
    
    
class Flatten(nn.Module):
    """Converts N-dimensional tensor into 'flat' one."""

    def __init__(self, keep_batch_dim=True):
        super().__init__()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x):
        if self.keep_batch_dim:
            return x.view(x.size(0), -1)
        return x.view(-1)
    
    
class CNN_Advanced(nn.Module):
    def __init__(self, raw_ni, fft_ni, raw_pos_ni, fft_pos_ni, no, drop=.5):
        super().__init__()
        
        self.raw = nn.Sequential(
            SepConv1d(raw_ni,  32, 8, 2, 3, drop=drop),
            SepConv1d(    32,  64, 8, 4, 2, drop=drop),
            SepConv1d(    64, 128, 8, 4, 2, drop=drop),
            SepConv1d(   128, 256, 8, 4, 2),
            Flatten(),
            nn.Dropout(drop), nn.Linear(2816, 64), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear( 64, 64), nn.ReLU(inplace=True),
            nn.BatchNorm1d(64)) #used to be BatchNormalization()
        
        self.raw_pos = nn.Sequential(
            SepConv1d(raw_pos_ni,  32, 4, 2, 3, drop=drop),
            SepConv1d(    32,  64, 4, 4, 2, drop=drop),
            SepConv1d(    64, 128, 8, 4, 2, drop=drop),
            SepConv1d(   128, 256, 8, 4, 2),
            Flatten(),
            nn.Dropout(drop), nn.Linear(512, 64), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear( 64, 16), nn.ReLU(inplace=True),
            nn.BatchNorm1d(16))

        
        self.fft = nn.Sequential(
            SepConv1d(fft_ni,  32, 8, 2, 4, drop=drop),
            SepConv1d(    32,  64, 8, 2, 4, drop=drop),
            SepConv1d(    64, 128, 8, 4, 4, drop=drop),
            SepConv1d(   128, 128, 8, 4, 4, drop=drop),
            SepConv1d(   128, 256, 8, 2, 3),
            Flatten(),
            nn.Dropout(drop), nn.Linear(3072, 64), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear( 64, 64), nn.ReLU(inplace=True),
            nn.BatchNorm1d(64))
        
        self.fft_pos = nn.Sequential(
            SepConv1d(fft_pos_ni,  32, 4, 2, 4, drop=drop),
            SepConv1d(    32,  64, 4, 2, 4, drop=drop),
            SepConv1d(    64, 128, 8, 4, 4, drop=drop),
            SepConv1d(   128, 128, 8, 4, 4, drop=drop),
            SepConv1d(   128, 256, 8, 2, 3),
            Flatten(),
            nn.Dropout(drop), nn.Linear(768, 64), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear( 64, 16), nn.ReLU(inplace=True),
            nn.BatchNorm1d(16))

        self.out = nn.Sequential(
            nn.Linear(160, 64), nn.ReLU(inplace=True), nn.Linear(64, no))
        
    def forward(self, features):
        t_raw, t_fft, t_raw_pos, t_fft_pos = features
        #print('bp1')
        raw_out = self.raw(t_raw)
       # print('bp2')
        fft_out = self.fft(t_fft)
       # print('bp3')
        #print(t_raw_pos.shape)
        raw_pos_out = self.raw_pos(t_raw_pos)
        
        fft_pos_out = self.fft_pos(t_fft_pos)
        
        #t_in = torch.cat([raw_out, fft_out], dim=1)
        t_in = torch.cat([raw_out, fft_out, raw_pos_out, fft_pos_out], dim=1)
       # print('bp4')
        out = self.out(t_in)
        return out
