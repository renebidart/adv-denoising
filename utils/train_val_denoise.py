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


# Based off: https://github.com/pytorch/examples/blob/master/imagenet/main.py
def train_epoch_denoise(lr, train_loader, denoise_model, requires_control, optimizer, epoch, device):
    """Train for 1 epoch. Is it weird to operate on stuff(loss etc.) without returning it?"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    orig_acc = AverageMeter()
    adv_acc = AverageMeter()

    denoise_model.train()
    end = time.time()
    
    for i, (orig, adv, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        orig, adv, target = orig.to(device), adv.to(device), target.to(device)

        if not requires_control:
            orig_pred, adv_pred, loss = denoise_model(orig, adv, requires_control = False)
        else:
            orig_pred, adv_pred, loss, closs = denoise_model(orig, adv, requires_control = True)

        # orig_acc.append(float(torch.sum(orig_pred.cpu().max() == target.cpu())/len(label)))
        # adv_acc.append(float(torch.sum(adv_pred.cpu().max() == target.cpu())/len(label)))

        curr_orig_acc, prec5 = accuracy(orig_pred, target, topk=(1, 5))
        curr_adv_acc, prec5 = accuracy(adv_pred, target, topk=(1, 5))
        orig_acc.update(curr_orig_acc[0], target.size(0))
        adv_acc.update(curr_adv_acc[0], target.size(0))
        losses.update(loss, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if requires_control:
        print(f'{epoch}: Original Acc {orig_acc.avg:.3f} \t Adversarial Acc  {adv_acc.avg:.3f} Loss ({losses.avg:.4f})\t'
              f'Time ({batch_time.avg:.3f})\t Data Time ({data_time.avg:.3f})  Control Loss {closs.item():.3f} ')

    else:
        print(f'{epoch}: Original Acc {orig_acc.avg:.3f} \t Adversarial Acc  {adv_acc.avg:.3f} Loss ({losses.avg:.4f})\t'
              f'Time ({batch_time.avg:.3f})\t Data Time ({data_time.avg:.3f})')
    return adv_acc, losses.avg



def validate_epoc_denoise(val_loader, denoise_model, requires_control, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    orig_acc = AverageMeter()
    adv_acc = AverageMeter()

    # switch to evaluate mode
    denoise_model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (orig, adv, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            
            orig, adv, target = orig.to(device), adv.to(device), target.to(device)

            if not requires_control:
                orig_pred, adv_pred, loss = denoise_model(orig, adv, requires_control = False)
            else:
                orig_pred, adv_pred, loss, closs = denoise_model(orig, adv, requires_control = True)

            curr_orig_acc, prec5 = accuracy(orig_pred, target, topk=(1, 5))
            curr_adv_acc, prec5 = accuracy(adv_pred, target, topk=(1, 5))
            orig_acc.update(curr_orig_acc[0], target.size(0))
            adv_acc.update(curr_adv_acc[0], target.size(0))
            losses.update(loss, target.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if requires_control:
            print(f'Original Acc {orig_acc.avg:.3f} \t Adversarial Acc {adv_acc.avg:.3f} Loss ({losses.avg:.4f})\t'
                  f'Time ({batch_time.avg:.3f})\t Data Time ({data_time.avg:.3f}   Control Loss {closs.item():.3f}')
        else:
            print(f'Original Acc {orig_acc.avg:.3f} \t Adversarial Acc  {adv_acc.avg:.3f} Loss ({losses.avg:.4f})\t'
                  f'Time ({batch_time.avg:.3f})\t Data Time ({data_time.avg:.3f})')
        return adv_acc, losses.avg


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
