import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

sys.path.insert(0,'/media/rene/code/foolbox')
import foolbox
from foolbox.attacks import FGSM, SinglePixelAttack, BoundaryAttack, LBFGSAttack

from models.cifar import ResNet, VGG, Wide_ResNet
from models import DenoiseHGD, UNet


def load_net_cifar(model_loc):
    """ Make a model
    Network must be saved in the form model_name-depth, where this is a unique identifier
    """
    model_file = Path(model_loc).name.rsplit('_')[0]
    model_name = model_file.split('-')[0]
    print('Loading model_file', model_file)
    if (model_name == 'lenet'):
        model = LeNet(10, 32)
    elif (model_name == 'vggnet'):
        model = VGG(int(model_file.split('-')[1]), 10, 32)
    elif (model_name == 'resnet'):
        model = ResNet(int(model_file.split('-')[1]), 10, 32)
    elif (model_name == 'wide'):
        model = Wide_ResNet(model_file.split('-')[2][0:2], model_file.split('-')[2][2:4], 0, 10, 32)
    else:
        print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')
        sys.exit(0)
    model.load_state_dict(torch.load(model_loc)['state_dict'])
    return model


# Return network & a unique file name
def net_from_args(args, num_classes, IM_SIZE):
    if (args.net_type == 'lenet'):
        net = LeNet(num_classes, IM_SIZE)
        file_name = 'lenet'
    elif (args.net_type == 'vggnet'):
        net = VGG(args.depth, num_classes, IM_SIZE)
        file_name = 'vgg-'+str(args.depth)
    elif (args.net_type == 'resnet'):
        net = ResNet(args.depth, num_classes, IM_SIZE)
        file_name = 'resnet-'+str(args.depth)
    elif (args.net_type == 'wide-resnet'):
        net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes, IM_SIZE)
        file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)
    elif (args.net_type == 'hgd'):
        fwd_out = [64, 128, 256, 256, 256]
        num_fwd = [2, 3, 3, 3, 3]
        back_out = [64, 128, 256, 256]
        num_back = [2, 3, 3, 3]
        fwd_in = 3
        net = DenoiseHGD(32, 32, fwd_in, fwd_out, num_fwd, back_out, num_back)
        file_name = 'hgd'
    elif (args.net_type == 'unet'):
        net = UNet(3, 3, args.stochastic)
        if args.stochastic:
            file_name = 'unet_stochastic'
        else:
            file_name = 'unet_stochastic'
    else:
        print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')
        sys.exit(0)
    return net, file_name


def denoise_from_args(args, IM_SIZE):
    if (args.denoise_type == 'hgd'):
        fwd_out = [64, 128, 256, 256, 256]
        num_fwd = [2, 3, 3, 3, 3]
        back_out = [64, 128, 256, 256]
        num_back = [2, 3, 3, 3]
        fwd_in = 3
        net = DenoiseHGD(IM_SIZE, IM_SIZE, fwd_in, fwd_out, num_fwd, back_out, num_back)
        file_name = 'hgd'
    elif (args.denoise_type == 'unet'):
        net = UNet(3, 3)
        file_name = 'unet'
    else:
        print('Error : Denoiser should be either [hgd / unet ')
        sys.exit(0)
    return net, file_name

def get_attack(attack_type, fmodel):
    if (attack_type == 'FGSM'):
        attack  = foolbox.attacks.FGSM(fmodel)
    elif (attack_type == 'SinglePixelAttack'):
        attack  = foolbox.attacks.SinglePixelAttack(fmodel)
    elif (attack_type == 'boundary'):
        attack  = foolbox.attacks.BoundaryAttack(fmodel)
    elif (attack_type == 'lbfgs'):
        attack  = foolbox.attacks.LBFGSAttack(fmodel)
    else:
        print('Error: Invalid attack_type')
        sys.exit(0)
    return attack