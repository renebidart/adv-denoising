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

with torch.cuda.device(0):
    from models.cifar import resnet
    from utils.data import make_generators_DF_cifar
    from utils.train_val_denoise import train_epoch_denoise, validate_epoc_denoise, save_checkpoint
    from utils.loading import load_net_cifar, denoise_from_args
    from models import DenoiseNet, DenoiseHGD, DenoiseLoss, UNet


    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--MODEL_SAVE_PATH', type=str)
    parser.add_argument('--files_df_loc', type=str)
    parser.add_argument('--device', type=str)

    parser.add_argument('--model_loc', type=str)
    parser.add_argument('--denoise_type', type=str)
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--lr', default=0.001, type=float, help='learning_rate')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    args = parser.parse_args()

    print('args.stochastic', args.stochastic)


    def main(args):
        epochs, batch_size, lr, num_workers = int(args.epochs), int(args.batch_size), float(args.lr),  int(args.num_workers)
        device = torch.device(args.device)

        MODEL_SAVE_PATH = Path(args.MODEL_SAVE_PATH)
        MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

        with open(args.files_df_loc, 'rb') as f:
            files_df = pickle.load(f)

        dataloaders = make_generators_DF_cifar(files_df, batch_size, num_workers, size=32, 
                                               path_colname='path', adv_path_colname='adv_path', return_loc=False)

        # LOAD EVERYTHING:
        classifier = load_net_cifar(args.model_loc).to(device)
        for p in classifier.parameters():
            p.requires_grad = False

        denoiser, model_name = denoise_from_args(args, IM_SIZE=32)
        denoiser = denoiser.to(device)

        loss = DenoiseLoss(n=1, hard_mining=0, norm=False)
        model = DenoiseNet(classifer=classifier, denoiser=denoiser, loss=loss).to(device)
        print('loaded classifier, denoiser, DenoiseNet')

        # their default optimizer (but they use batch_size of 60)
        optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = 0.0001) # using paper init, not the code
        base_lr = lr

        best_val_loss = 1000000
        metrics = {}
        metrics['train_adv_acc'] = []
        metrics['train_loss'] = []
        metrics['val_adv_acc'] = []
        metrics['val_loss'] = []

        def get_lr(curr_epoch, epochs, base_lr):
            if epoch <= epochs * 0.6:
                return base_lr
            elif epoch <= epochs * 0.9:
                return base_lr * 0.1
            else:
                return base_lr * 0.01

        for epoch in range(epochs):
            requires_control = epoch == 0
            # set learning rate
            lr = get_lr(epoch, epochs, base_lr)
            for param_group in optimizer.param_groups: # why this way?
                param_group['lr'] = lr

            # train for one epoch
            train_adv_acc, train_loss = train_epoch_denoise(lr, dataloaders['train'], model, requires_control, optimizer, epoch, device)
            metrics['train_adv_acc'].append(train_adv_acc)
            metrics['train_loss'].append(train_loss)

            # evaluate on validation set
            val_adv_acc, val_loss = validate_epoc_denoise(dataloaders['val'], model, requires_control, device)
            metrics['val_adv_acc'].append(val_adv_acc)
            metrics['val_loss'].append(val_loss)
            
            # remember best loss and save checkpoint
            is_best = val_loss < best_val_loss
            best_val_loss = min(val_loss, best_val_loss)
            save_checkpoint({
                'model_name': model_name,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss,
                'metrics': metrics,
            }, is_best, model_name, MODEL_SAVE_PATH)

    if __name__ == '__main__':
        main(args)


