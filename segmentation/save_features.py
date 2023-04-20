import functools
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np

from segmentation.data_utils.P2NetDataset import SavedP2NetTraining

from Constants.constants import ROOT_DIR
import os

from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)





def save_features():
    # pytorch optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # creating path to save features for training
    features_dir = os.path.join(ROOT_DIR,'segmentation','Saved_Features')

    npoints = 50000
    num_classes = 19
    num_seq = 3
    num_epochs = 1
    train_batch_size = 64

    saved_preds_path = os.path.join(ROOT_DIR,'segmentation','Saved_Preds')
    train_set = SavedP2NetTraining(saved_preds_path)

    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=False)



    for epoch in range(num_epochs):
        for i, (features, labels, save_pred_file) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
            for i in range(train_batch_size):
                save_obj = {
                    'features': features[i].clone(),
                    'labels': labels[i].clone(),
                }
                file = save_pred_file[i]
                # getting sequence number
                sequence = file.parent.parent.name
                # getting the frame number  without the '.bin'
                frame_number = file.parent.name

                save_dir = os.path.join(features_dir, sequence)

                # check if this dir exists
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)

                save_dir = os.path.join(save_dir, frame_number)

                # check if this dir exists
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)

                # save to .pt torch file
                torch.save(save_obj, os.path.join(save_dir, str(epoch + 1) + '.pt'))