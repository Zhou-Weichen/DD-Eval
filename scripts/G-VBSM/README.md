
'base on G_VBSM_Dataset_Condensation/'
mkdir squeeze_Models
cd squeeze_Models

wget https://github.com/shaoshitong/G_VBSM_Dataset_Condensation/releases/download/v0.0.1/squeeze_wo_ema_cifar10.zip
wget https://github.com/shaoshitong/G_VBSM_Dataset_Condensation/releases/download/v0.0.1/squeeze_wo_ema_cifar100.zip
wget https://github.com/shaoshitong/G_VBSM_Dataset_Condensation/releases/download/v0.0.1/squeeze_wo_ema_tiny_imagenet.zip
unzip squeeze_wo_ema_cifar10.zip
unzip squeeze_wo_ema_cifar100.zip
unzip squeeze_wo_ema_tiny_imagenet.zip

'base on G_VBSM_Dataset_Condensation/'
mkdir recover_Data
cd recover_Data

download data from "https://github.com/shaoshitong/G_VBSM_Dataset_Condensation" [Open Distilled Datasets]