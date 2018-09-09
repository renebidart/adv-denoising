import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import torchvision.transforms as T
from torchvision.models import resnet18, vgg16
from torch.optim import lr_scheduler
import torch.optim as optim
import time
import shutil


def train_model(model, model_name, SAVE_PATH, criterion, optimizer, scheduler, epochs, dataloaders, device):
    print('Using device:',  device)
    model = model.to(device)
    best_val_acc = 0
    metrics = {}
    metrics['train_top1_acc'] = []
    metrics['train_losses'] = []
    metrics['val_top1_acc'] = []
    metrics['val_losses'] = []
    
    for epoch in range(epochs):
        # train for one epoch
        train_top1_acc, train_losses = train_epoch(dataloaders['train'], model, criterion, optimizer, epoch, device)
        metrics['train_top1_acc'].append(train_top1_acc)
        metrics['train_losses'].append(train_losses)

        # evaluate on validation set
        val_top1_acc, val_losses = validate_epoch(dataloaders['val'], model, criterion, device)
        metrics['val_top1_acc'].append(val_top1_acc)
        metrics['val_losses'].append(val_losses)
        
        #lr scheduler
        scheduler.step()

        # remember best validation accuracy and save checkpoint
        is_best = val_top1_acc > best_val_acc
        best_val_acc = max(val_top1_acc, best_val_acc)
        save_checkpoint({
            'model_name': model_name,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'epoch': epoch + 1,
            'best_val_acc': best_val_acc,
            'metrics': metrics,
        }, is_best, model_name, SAVE_PATH)


# Based off: https://github.com/pytorch/examples/blob/master/imagenet/main.py
def train_epoch(train_loader, model, criterion, optimizer, epoch, device):
    """Train for 1 epoch. Is it weird to operate on stuff(loss etc.) without returning it?"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()
    
    for i, (inputs, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        inputs, target = inputs.to(device), target.to(device)
        output = model(inputs)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    print(f'{epoch}:     * TOP1 {top1.avg:.3f} TOP5 {top5.avg:.3f} Loss ({losses.avg:.4f})\t'
          f'Time ({batch_time.avg:.3f})\t Data Time ({data_time.avg:.3f})')
    return top1.avg, losses.avg

def validate_epoch(val_loader, model, criterion, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(val_loader):
            inputs, target = batch[0].to(device), batch[1].to(device)
            output = model(inputs)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
    print(f'VALID:  * TOP1 {top1.avg:.3f} TOP5 {top5.avg:.3f} Loss ({losses.avg:.4f})\t',
          f'Time ({batch_time.avg:.3f})\t')
    return top1.avg, losses.avg


def save_checkpoint(state, is_best, model_name, PATH):
    save_path = str(PATH)+'/'+str(model_name)+'_ckpnt.pth.tar'
    torch.save(state, save_path)
    if is_best:
        best_path = str(PATH)+'/'+str(model_name)+'_model_best.pth.tar'
        shutil.copyfile(save_path, best_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred).type(torch.cuda.LongTensor))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
