# Denoising Adversarial Images


## Denoising is mostly useless for adversarial attacks
Any standard denoiser can just be attacked as a normal model by treating the denoiser+model as a new model. Adding stochastic pooling into this doesn't help. My guess is that generative models are the only way.


## How to get it going:
### Setup
Create a virtualenv to install everything in and activate:

```
python3 -m venv ENV
source ENV/bin/activate
```

Install requirements:

```
pip install -r requirements.txt
```

### Download CIFAR and put it in a nicer format
Downloads original dataset to PATH, and stores the nicer format in NEW\_PATH

```
python make_cifar_normal.py --PATH .../data --NEW_PATH ...cifar10_normal 
```

### Train a network on CIFAR
DATA\_PATH is path to original CIFAR dataset, save\_path is folder where model is saved

```
python train_net_cifar.py --DATA_PATH .../data --SAVE_PATH ...cifar10_normal/models  --net_type resnet --depth 50 --dropout 0 --device cuda:1
```

### Make Adversarial Dataset
files\_df\_loc is a dict of pandas dataframes for the train and test sets, NEW\_PATH is where to store adversarial dataset, model\_loc is the location of model used to create adversarial examples

```
python make_adv_data.py --files_df_loc .../cifar10_normal/files_df.pkl --NEW_PATH .../adv_fgsm/resnet50/ --model_loc ...cifar10_normal/models/resnet-50_model_best.pth.tar --attack_type FGSM --device cuda:0
```

### Train a Denoiser
MODEL\_SAVE\_PATH is where to save the denoiser, files\_df\_loc should contain both adversarial and normal locations, model\_loc  is the classifier being used,

```
python train_denoiser_cifar.py --MODEL_SAVE_PATH .../adv_fgsm/resnet50/sample/models --files_df_loc .../adv_fgsm/resnet50/sample/files_df_adv.pkl --model_loc .../cifar10_normal/models/resnet-50_model_best.pth.tar ----denoise_type unet --batch_size 64 --epochs 30 --stochastic --device cuda:1
```