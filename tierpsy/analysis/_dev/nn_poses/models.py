#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 14:25:26 2019

@author: avelinojaver
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def conv_layer(ni, nf, ks=3, stride=1, dilation=1):
    if isinstance(ks, (float, int)):
        ks = (ks, ks)
    
    if isinstance(dilation, (float, int)):
        dilation = (dilation, dilation)
    
    pad = [x[0]//2*x[1] for x in zip(ks, dilation)]
    
    return nn.Sequential(
           nn.Conv2d(ni, nf, ks, bias = False, stride = stride, padding = pad, dilation = dilation),
           nn.BatchNorm2d(nf),
           nn.ReLU(inplace = True)
           )
    
    
def _conv3x3(n_in, n_out):
    return [nn.Conv2d(n_in, n_out, 3, padding=1),
    nn.LeakyReLU(negative_slope=0.1, inplace=True)]

class CPM(nn.Module):
    '''
    Convolutional pose machine
    https://arxiv.org/abs/1602.00134
    '''
    def __init__(self, 
                 n_stages = 6, 
                 n_segments = 49,
                 same_output_size = False):
        
        super().__init__()
        
        self.n_segments = n_segments
        self.n_stages = n_stages
        self.same_output_size = same_output_size
        
        
        self.feats = nn.Sequential(*[
            conv_layer(1, 64, ks=3, stride = 1),
            conv_layer(64, 64, ks=3, stride = 1),
            conv_layer(64, 128, ks=3, stride = 2),
            conv_layer(128, 128, ks=3, stride = 1),
            conv_layer(128, 256, ks=3, stride = 2),
            conv_layer(256, 256, ks=3, stride = 1),
            conv_layer(256, 256, ks=3, stride = 1),
        ])
        self.n_input_feats = 256
        
        mm = self.n_input_feats
        self.stage1 = nn.Sequential(*[
            conv_layer(mm, mm, ks=3, stride = 1),
            conv_layer(mm, mm, ks=3, stride = 1),
            conv_layer(mm, mm, ks=3, stride = 1),
            conv_layer(mm, mm//2, ks=1, stride = 1),
            conv_layer(mm//2, self.n_segments, ks=1, stride = 1)
        ])
    
        
        self.stages = []
        for ii in range(1, self.n_stages):
            new_stage = nn.Sequential(*[
                conv_layer(mm + self.n_segments, mm, ks=3, stride = 1),
                conv_layer(mm, mm, ks=3, stride = 1),
                conv_layer(mm, mm, ks=3, stride = 1),
                conv_layer(mm, mm, ks=3, stride = 1),
                conv_layer(mm, mm, ks=3, stride = 1),
                conv_layer(mm, mm//2, ks=1, stride = 1),
                conv_layer(mm//2, self.n_segments, ks=1, stride = 1)
            ])
    
    
            
            stage_name = 'stage{}'.format(ii+1)
            setattr(self, stage_name, new_stage)        
    
            self.stages.append(new_stage)
        
        if self.same_output_size:
            self.upscaling = nn.Sequential(*[
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    *_conv3x3(self.n_segments, self.n_segments),
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    *_conv3x3(self.n_segments, self.n_segments)
                ])
        
    def forward(self, X):
        if self.same_output_size:
            nn = 2**2
            ss = [math.ceil(x/nn)*nn - x for x in X.shape[2:]]
            pad_ = [(int(math.floor(x/2)),int(math.ceil(x/2))) for x in ss]
            
            #use pytorch for the padding
            pad_ = [x for d in pad_[::-1] for x in d] 
            pad_inv_ = [-x for x in pad_] 
            X = F.pad(X, pad_, 'reflect')
        
        feats = self.feats(X)
        
        
        x_out = self.stage1(feats)
        
        outs = [x_out]
        for stage in self.stages:
            x_in = torch.cat([feats, x_out], dim=1)
            x_out = stage(x_in)
            outs.append(x_out)
        
        if self.same_output_size:
            outs = [self.upscaling(x) for x in outs]
            outs = [F.pad(x, pad_inv_) for x in outs]
            
        return outs
        
        
        
class CPMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2 = nn.MSELoss()        
        
        
    def forward(self, outs, target):
        
        loss = self.l2(outs[0], target)
        for x in outs[1:]:
            loss.add_(self.l2(x, target))
        
        return loss
#%%
def _stage_stacked_convs(n_in, n_out, n_core):
    _ini_layer = [conv_layer(n_in, n_core, ks=3, stride = 1)]
    _rep_layers = [conv_layer(n_core, n_core, ks=3, stride = 1) for _ in range(4)]
    _out_layers = [conv_layer(n_core, n_core//2, ks=1, stride = 1),
                conv_layer(n_core//2, n_out, ks=1, stride = 1)]
    
    return  _ini_layer + _rep_layers + _out_layers

def _first_stacked_convs(n_in, n_out, n_core):
     return [
            conv_layer(n_in, n_core, ks=3, stride = 1),
            conv_layer(n_core, n_core, ks=3, stride = 1),
            conv_layer(n_core, n_core, ks=3, stride = 1),
            conv_layer(n_core, n_core//2, ks=1, stride = 1),
            conv_layer(n_core//2, n_out, ks=1, stride = 1)
        ]
     
def _upscaling_convs(n_in):
    return [nn.Upsample(scale_factor=2, mode='bilinear'),
            *_conv3x3(n_in, n_in),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            *_conv3x3(n_in, n_in)
            ]
        
class CPM_PAF(nn.Module):
    '''
    Convolutional pose machine + Part Affinity Fields
    https://arxiv.org/abs/1602.00134
    https://arxiv.org/pdf/1611.08050.pdf
    '''
    def __init__(self, 
                 n_stages = 6, 
                 n_segments = 25,
                 n_affinity_maps = 4,
                 same_output_size = False):
        
        super().__init__()
        
        self.n_segments = n_segments
        self.n_affinity_maps = n_affinity_maps
        self.n_stages = n_stages
        self.same_output_size = same_output_size
        
        self.n_input_feats = 256
        
        self.feats = nn.Sequential(*[
            conv_layer(1, 64, ks=3, stride = 1),
            conv_layer(64, 64, ks=3, stride = 1),
            conv_layer(64, 128, ks=3, stride = 2),
            conv_layer(128, 128, ks=3, stride = 1),
            conv_layer(128, 256, ks=3, stride = 2),
            conv_layer(256, 256, ks=3, stride = 1),
            conv_layer(256, 256, ks=3, stride = 1),
        ])
        
        
        _layers = _first_stacked_convs(n_in = self.n_input_feats, 
                                       n_out = self.n_segments, 
                                       n_core = 256)
        self.CPM_stage1 = nn.Sequential(*_layers)
        
        _layers = _first_stacked_convs(n_in = self.n_input_feats, 
                                       n_out = self.n_affinity_maps*2, 
                                       n_core = 256)
        self.PAF_stage1 = nn.Sequential(*_layers)
        
        
        n_stage_inputs = self.n_input_feats + self.n_segments + self.n_affinity_maps*2
        
        self.CPM_stages = []
        for ii in range(1, self.n_stages):
            _layers = _stage_stacked_convs(n_in = n_stage_inputs, 
                                           n_out = self.n_segments,
                                           n_core = 256)
            
            new_stage = nn.Sequential(*_layers)
            stage_name = 'CPM_stage{}'.format(ii+1)
            setattr(self, stage_name, new_stage)        
    
            self.CPM_stages.append(new_stage)
        
        self.PAF_stages = []
        for ii in range(1, self.n_stages):
            _layers = _stage_stacked_convs(n_in = n_stage_inputs, 
                                           n_out = self.n_affinity_maps*2,
                                           n_core = 256)
            
            new_stage = nn.Sequential(*_layers)
            stage_name = 'PAF_stage{}'.format(ii+1)
            setattr(self, stage_name, new_stage)        
    
            self.PAF_stages.append(new_stage)
            
        
        
        if self.same_output_size:
            _layers = _upscaling_convs(n_in = self.n_segments)
            self.CPM_upscaling = nn.Sequential(*_layers)
            
            _layers = _upscaling_convs(n_in = self.n_affinity_maps*2)
            self.PAF_upscaling = nn.Sequential(*_layers)
        
    def forward(self, X):
        if self.same_output_size:
            nn = 2**2
            ss = [math.ceil(x/nn)*nn - x for x in X.shape[2:]]
            pad_ = [(int(math.floor(x/2)),int(math.ceil(x/2))) for x in ss]
            
            #use pytorch for the padding
            pad_ = [x for d in pad_[::-1] for x in d] 
            pad_inv_ = [-x for x in pad_] 
            X = F.pad(X, pad_, 'reflect')
        
        feats = self.feats(X)
        
        
        cpm_out = self.CPM_stage1(feats)
        paf_out = self.PAF_stage1(feats)
        
        outs = [(cpm_out, paf_out)]
        
        for (cpm_stage, paf_stage) in zip(self.CPM_stages, self.PAF_stages):
            x_in = torch.cat([feats, cpm_out, paf_out], dim=1)
            
            cpm_out = cpm_stage(x_in)
            paf_out = paf_stage(x_in)
            
            
            outs.append((cpm_out, paf_out))
        
        if self.same_output_size:
            outs_upscaled = []
            for (cpm_out, paf_out) in outs:
                cpm_out = self.CPM_upscaling(cpm_out)
                cpm_out = F.pad(cpm_out, pad_inv_)
                
                paf_out = self.PAF_upscaling(paf_out)
                paf_out = F.pad(paf_out, pad_inv_)
                
                outs_upscaled.append((cpm_out, paf_out))
            
            outs = outs_upscaled
        
        outs_n = []
        for (cpm_out, paf_out) in outs:
            _dims = paf_out.shape
            paf_out = paf_out.view(_dims[0], _dims[1]//2, 2, _dims[-2], _dims[-1])
            outs_n.append((cpm_out, paf_out))
        outs = outs_n
            
        return outs
    
class CPM_PAF_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2 = nn.MSELoss()        
        
        
    def forward(self, outs, target):
        cpm_target, paf_target = target
        cpm_out, paf_out = outs[0]
        
        cpm_loss = self.l2(cpm_out, cpm_target)
        paf_loss = self.l2(paf_out, paf_target)
        
        loss = cpm_loss + paf_loss
        for (cpm_out, paf_out) in outs[1:]:
            cpm_loss = self.l2(cpm_out, cpm_target)
            paf_loss = self.l2(paf_out, paf_target)
            loss.add_(cpm_loss + paf_loss)
            
        
        return loss
#%%
if __name__ == '__main__':
    
    X = torch.rand([4, 1, 162, 135])
    target = (torch.rand([4, 25, 162, 135]), torch.rand([4, 4, 2, 162, 135]))
    
    mod = CPM_PAF(same_output_size = True)
    criterion = CPM_PAF_Loss()
    
    
    outs = mod(X)
    loss = criterion(outs, target)
    loss.backward()