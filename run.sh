python train_denoiser_cifar.py --MODEL_SAVE_PATH /media/rene/data/adv_denoising/cifar10/adv_one/resnet50/sample/models --files_df_loc /media/rene/data/adv_denoising/cifar10/adv_one/resnet50/sample/files_df_adv.pkl --model_loc /media/rene/data/adv_denoising/cifar10/cifar10_normal/models/resnet-50_model_best.pth.tar --denoise_type unet --batch_size 100 --epochs 30 --device cuda:0
python train_denoiser_cifar.py --MODEL_SAVE_PATH /media/rene/data/adv_denoising/cifar10/adv_one/resnet50/sample/models --files_df_loc /media/rene/data/adv_denoising/cifar10/adv_one/resnet50/sample/files_df_adv.pkl --model_loc /media/rene/data/adv_denoising/cifar10/cifar10_normal/models/resnet-50_model_best.pth.tar --denoise_type unet --batch_size 100 --epochs 30 --stochastic --device cuda:0

python make_adv_data.py --files_df_loc /media/rene/data/adv_denoising/cifar10/cifar10_normal/sample_df.pkl --NEW_PATH /media/rene/data/adv_denoising/cifar10/adv_one/resnet50_denoised/sample --model_loc /media/rene/data/adv_denoising/cifar10/adv_one/resnet50/sample/models/unet_model_best.pth.tar --attack_type SinglePixelAttack --device cuda:0
python make_adv_data.py --files_df_loc /media/rene/data/adv_denoising/cifar10/cifar10_normal/sample_df.pkl --NEW_PATH /media/rene/data/adv_denoising/cifar10/adv_one/resnet50_stoch_denoised/sample --model_loc /media/rene/data/adv_denoising/cifar10/adv_one/resnet50/sample/models/unet_stochastic_model_best.pth.tar --attack_type SinglePixelAttack --device cuda:0