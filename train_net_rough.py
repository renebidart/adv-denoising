"""
Training for baselines for cifar. 
Use nets from https://github.com/meliketoy/wide-resnet.pytorch (lenet, vgg, resnet, wide-resnet)
Not optimal way to train, using dumb step lr, etc.

"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.optim import lr_scheduler

# Add the src directory and import my functions
src_dir = Path.cwd().parents[1] / 'src'
sys.path.append(src_dir)
from networks import ResNet
from utils.data import make_gen_std_cifar
from utils.train_val import*
from utils.loading import make_net_cifar


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--DATA_PATH', type=str)
parser.add_argument('--SAVE_PATH', type=str)
parser.add_argument('--files_df_loc', type=str)
parser.add_argument('--device', type=str)

# Defining the network:
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
# training params
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--epochs', default=350, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--im_size', default=32, type=int)
parser.add_argument('--train_cifar', action='store_true', help='Train on standard cifar dataset')
args = parser.parse_args()


def main(args):
    epochs, batch_size, files_df_loc, lr = int(args.epochs), int(args.batch_size), str(args.files_df_loc), float(args.lr)
    im_size, num_workers, train_cifar = int(args.im_size), int(args.num_workers), str(args.train_cifar)
    device = torch.device(args.device)

    PATH = Path(args.DATA_PATH)
    SAVE_PATH = Path(args.SAVE_PATH)
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    # Make generators, get dataset info:
    if args.train_cifar:
        files_df = ''
        num_classes = 10
    else:
        with open(files_df_loc, 'rb') as f:
            files_df = pickle.load(f)
        num_classes = files_df.iloc[:, 1].nunique()
    dataloaders = make_gen_std_cifar(PATH, batch_size, num_workers, train_cifar, files_df, im_size)

    # get the network
    model, model_name = make_net_cifar(SAVE_PATH)
    print(f'--------- Training: {model_name} with depth {args.depth} ---------')

    # get training parameters and train:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs/3), gamma=0.1) # close enough
    train_model(model, model_name, SAVE_PATH, criterion, optimizer, scheduler, epochs, dataloaders, device=device)

if __name__ == '__main__':
    main(args)
