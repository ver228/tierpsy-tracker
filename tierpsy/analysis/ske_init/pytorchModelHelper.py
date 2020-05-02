#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 14:07:26 2020

@author: lferiani

definition of the pytorch model as implemented by Ziwei
https://github.com/wylhtydtm/Nematode-project
"""

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

# %% model class definition


class ConvNet(nn.Module):
    # Class : 0= non-worm+ aggregates worms ; 1: valid+difficult
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),  # activation layer
                # conv layer taking the output of the previous layer:
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))  # activation layer

        self.drop_out = nn.Dropout2d(0.5)
        # define fully connected layer:
        self.fc_layers = nn.Sequential(nn.Linear(512*5*5, 2))

    def forward(self, x):
        x = self.conv_layers(x)  # pass input through conv layers
        x = self.drop_out(x)
        # flatten output for fully connected layer, batchize,
        # -1 do whatever it needs to be
        x = x.view(x.shape[0], -1)
        x = self.fc_layers(x)  # pass  through fully connected layer
        # softmax activation function on outputs,
        # get probability distribution on output, all ouputs add to 1
        x = F.softmax(x, dim=1)
        return x


def prep_for_pytorch(img):
    if img.ndim == 3:
        img = img - img.min(axis=(1, 2))[:, None, None]
        img = img / img.max(axis=(1, 2))[:, None, None]
        img = np.expand_dims(img, axis=1).astype(np.float32)
    else:
        img = img - img.min()
        img = img / img.max()
        img = img.astype(np.float32)[None, None, :, :]
    return img


def predict_image(image):
    """This is not used by tierpsy, but good for testing and debugging"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = torch.tensor(image).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        proba = output.data.numpy()[:, 1]
    index = output.data.cpu().numpy().argmax(axis=1)
    # return index, output
    return proba, index


def load_pytorch_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNet().to(device)  # to instantiate model
    model.load_state_dict(torch.load(path,
                                     map_location=device))
    model.eval()
    return model


# model was trained on 80-by-80 px size ROIs, let's write that down
MODEL_ROIS_TRAINED_SIZE = 80


# %% MAIN
if __name__ == '__main__':

    # %%% imports

    import pandas as pd
    from filterTrajectModel import shift_and_normalize

    # %%% paths

    dataset_path = ('/Users/lferiani/OneDrive - Imperial College London/'
                    + 'Analysis/Ziweis_NN/Hydra_dataset/worm_ROI_samples.hdf5')
    labels_path = ('/Users/lferiani/OneDrive - Imperial College London/'
                   + 'Analysis/Ziweis_NN/Hydra_dataset/'
                   + 'worm_ROI_samples_annotationsonly.hdf5')
    # path to the saved model_state
    model_path = ('/Users/lferiani/OneDrive - Imperial College London/'
                  + 'Analysis/Ziweis_NN/Model/'
                  + 'notaworm_vs_worm_difficult_aggregate_model_state.pth')

    # %%% read images and labels

    n_imgs = 1000

    with pd.HDFStore(labels_path, 'r') as fid:
        labels = fid['/sample_data'].copy()
    labels = labels.query(f"resampled_index_z < {n_imgs}")

    img_ind = (0, n_imgs+1)
    img = np.zeros((n_imgs, 80, 80))
    with pd.HDFStore(dataset_path, 'r') as fid:
        cc = 0
        for _, row in labels.iterrows():
            this_img = fid.get_node('/mask').read(int(row['img_row_id']))
            this_img = this_img[0, 40:120, 40:120]
            img[cc] = this_img
            cc += 1

    # %%% preprocess

    img = shift_and_normalize(img)
    img = prep_for_pytorch(img)

    # %%% load model and predict

    model = load_pytorch_model(model_path)
    probs, _ = predict_image(img)
    preds = probs > 0.5

    # %%% measure performance (i.e. check I implemented it right)
    # at the moment, this model predicts bad for bad + aggregate (0 and 4)
    # good for single and difficult (2 and 3)

    manually_good_worm = np.logical_and(labels['label_id_z'].values > 1,
                                        labels['label_id_z'].values <= 4)

    true_pos = np.sum(np.logical_and(preds == 1, manually_good_worm == 1))
    false_pos = np.sum(np.logical_and(preds == 1, manually_good_worm == 0))
    false_neg = np.sum(np.logical_and(preds == 0, manually_good_worm == 1))

    accuracy = np.sum(preds == manually_good_worm) / n_imgs
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)

    print(f"accuracy = {accuracy}, precision = {precision}, recall = {recall}")
