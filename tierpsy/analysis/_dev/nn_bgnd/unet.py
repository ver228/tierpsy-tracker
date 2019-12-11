#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 11:13:46 2018

@author: avelinojaver

"""
import math
import torch
from torch import nn
import torch.nn.functional as F


def weights_init_xavier(m):
    '''
    Taken from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    '''
    classname = m.__class__.__name__
    # print(classname)
    if classname.startswith('Conv'):
        nn.init.xavier_normal_(m.weight.data, gain=1)
    elif classname.startswith('Linear'):
        nn.init.xavier_normal_(m.weight.data, gain=1)
    elif classname.startswith('BatchNorm2d'):
        nn.init.uniform_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def _crop(x, x_to_crop):
    c = (x_to_crop.size()[2] - x.size()[2])/2
    c1, c2 =  math.ceil(c), math.floor(c)
    
    c = (x_to_crop.size()[3] - x.size()[3])/2
    c3, c4 =  math.ceil(c), math.floor(c)
    
    cropped = F.pad(x_to_crop, (-c3, -c4, -c1, -c2)) #negative padding is the same as cropping
    
    return cropped
def _conv3x3(n_in, n_out):
    return [nn.Conv2d(n_in, n_out, 3, padding=1),
    nn.LeakyReLU(negative_slope=0.1, inplace=True)]

class Down(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        _layers = _conv3x3(n_in, n_out) + [nn.MaxPool2d(2)]
        self.conv_pooled = nn.Sequential(*_layers)

    def forward(self, x):
        x = self.conv_pooled(x)
        return x


class Up(nn.Module):
    def __init__(self, n_filters):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        _layers = []
        for ii in range(len(n_filters) - 1):
            n_in, n_out = n_filters[ii], n_filters[ii+1]
            _layers += _conv3x3(n_in, n_out)
        self.conv = nn.Sequential(*_layers)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = _crop(x1, x2)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels = 1, n_classes = 1):
        super().__init__()
        self.conv0 = nn.Sequential(*_conv3x3(n_channels, 48))
        
        self.down1 = Down(48, 48)
        self.down2 = Down(48, 48)
        self.down3 = Down(48, 48)
        self.down4 = Down(48, 48)
        self.down5 = Down(48, 48)
        
        self.conv6 = nn.Sequential(*_conv3x3(48, 48))
        
        self.up5 = Up([96, 96, 96])
        self.up4 = Up([144, 96, 96])
        self.up3 = Up([144, 96, 96])
        self.up2 = Up([144, 96, 96])
        self.up1 = Up([96 + n_channels, 64, 32])
        
        self.conv_out = nn.Sequential(nn.Conv2d(32, n_classes, 3, padding=1))
        
        for m in self.modules():
            weights_init_xavier(m)
    
    def _unet(self, x_input):    
        x0 = self.conv0(x_input)
        
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        
        x6 = self.conv6(x5)
        
        x = self.up5(x6, x4)
        x = self.up4(x, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x_input)
        
        x = self.conv_out(x)
        return x
    
    def forward(self, x_input):
        # the input shape must be divisible by 32 otherwise it will be cropped due 
        #to the way the upsampling in the network is done. Therefore it is better to path 
        #the image and recrop it to the original size
        nn = 2**5
        ss = [math.ceil(x/nn)*nn - x for x in x_input.shape[2:]]
        pad_ = [(int(math.floor(x/2)),int(math.ceil(x/2))) for x in ss]
        
        #use pytorch for the padding
        pad_ = [x for d in pad_[::-1] for x in d] 
        pad_inv_ = [-x for x in pad_] 
        
        
        x_input = F.pad(x_input, pad_, 'reflect')
        x = self._unet(x_input)
        x = F.pad(x, pad_inv_)
        
        
        return x


if __name__ == '__main__':
    mod = UNet()
    X = torch.rand((1, 1, 540, 600))
    out = mod(X)
    
    print(out.size())
     
    
