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
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler

from models.cifar import resnet
from utils.data import make_gen_std_cifar
from utils.train_val import train_model
from utils.loading import net_from_args


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--DATA_PATH', type=str)
parser.add_argument('--SAVE_PATH', type=str)
parser.add_argument('--device', type=str)

# Defining the network:
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0, type=float, help='dropout_rate')
# training params
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--epochs', default=350, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=4, type=int)
args = parser.parse_args()


def main(args):
    epochs, batch_size, lr, num_workers = int(args.epochs), int(args.batch_size), float(args.lr),  int(args.num_workers)
    device = torch.device(args.device)

    PATH = Path(args.DATA_PATH)
    SAVE_PATH = Path(args.SAVE_PATH)
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    # Make generators:
    dataloaders = make_gen_std_cifar(PATH, batch_size, num_workers)

    # get the network
    model, model_name = net_from_args(args, num_classes=10, IM_SIZE=32)
    print(f'--------- Training: {model_name} with depth {args.depth} ---------')

    # get training parameters and train:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs/3), gamma=0.1) # close enough
    train_model(model, model_name, SAVE_PATH, criterion, optimizer, scheduler, epochs, dataloaders, device=device)

if __name__ == '__main__':
    main(args)
