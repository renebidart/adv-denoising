"""
Based on the files_df_loc

make adv data from cifar
attack should be one of:
FGSM, SinglePixelAttack, 

"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# until the github pip foolbox is updated to use device properly
sys.path.insert(0,'/media/rene/code/foolbox')
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from PIL import Image
import foolbox

from utils.loading import load_net_cifar, get_attack
from utils.data import FilesDFImageDataset
from utils.display import load_image

parser = argparse.ArgumentParser()
parser.add_argument('--files_df_loc', type=str)
parser.add_argument('--NEW_PATH', type=str)
parser.add_argument('--model_loc', type=str)
parser.add_argument('--attack_type', type=str)
parser.add_argument('--device', type=str)
args = parser.parse_args()

files_df_loc, NEW_PATH, model_loc, attack_type, device = args.files_df_loc, Path(args.NEW_PATH), args.model_loc, args.attack_type, torch.device(args.device)


def make_adv_data_cifar(files_df_loc, NEW_PATH, model_loc, attack_type, device):
    num_workers = 0
    batch_size = 1

    new_files = {}
    with open(args.files_df_loc, 'rb') as f:
        files_df = pickle.load(f)

    mean = np.array([0.4914, 0.4822, 0.4465]).reshape((3, 1, 1))
    std = np.array([0.2023, 0.1994, 0.2010]).reshape((3, 1, 1))

    data_transforms = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    # data_transforms = None

    model = load_net_cifar(args.model_loc).to(device).eval()
    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10, preprocessing=(mean, std))
    attack  = get_attack(attack_type, fmodel)

    for folder_name, files in files_df.items():
        print('Making adversarial examples for ', folder_name)
        dataset = FilesDFImageDataset(files, data_transforms, return_loc=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        SAVE_PATH = NEW_PATH / folder_name
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
        new_files[folder_name] = pd.DataFrame()
        problem_examples = 0
        
        for i, batch in enumerate(tqdm(dataloader)):
            image, target, file_loc = batch[0].numpy(), int(batch[1].numpy()), batch[2][0]
            image = load_image(file_loc)
            # image = np.squeeze(image) / 1.
            adv_img = attack(image, target)

            if adv_img is None:
                print('adv_img is None')
                adv_img = image
            else:
                adv_img = np.moveaxis(np.squeeze(adv_img), 0, 2) * 255. # for whatever reason Pil likes this format
            try:
                adv_img = Image.fromarray(adv_img.astype('uint8'))
                adv_path = str(SAVE_PATH) + '/adv_'+file_loc.split('/')[-1]
                adv_img.save(adv_path)
                new_files[folder_name] = new_files[folder_name].append({'path': file_loc, 'class': np.squeeze(target), 
                                                                        'adv_path': adv_path, 'attack_type': attack_type}, ignore_index=True)
            except Exception as e:
                problem_examples +=1
                print(e)
                print('type(adv_img) ', type(adv_img))
                print('adv_img.shape ', adv_img.shape)
        print('problem_examples: ', problem_examples)


    with open(str(NEW_PATH)+'/files_df_adv.pkl', 'wb') as f:
        pickle.dump(new_files, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    make_adv_data_cifar(files_df_loc, NEW_PATH, model_loc, attack_type, device)