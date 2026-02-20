# wandb disabled
wandb enabled
wandb offline

CUDA_VISIBLE_DEVICES=0 python train_FKD.py \
    --wandb-project 'final_rn18_fkd' \
    --batch-size 64 \
    --model "ResNet18" \
    --cos --loss-type "mse_gt" --ce-weight 0.15 \
    -j 4 --gradient-accumulation-steps 1 \
    -T 20 --sgd --sgd-lr 0.1 --adamw-lr 0.001 \
    --mix-type 'cutmix' \
    --output-dir ./save/final_rn_fkd/ \
    --train-dir ../../recover_Data/GVBSM_CIFAR_100_Recover_IPC_50_1 \
    --val-dir /data4t2/ZHOU/G_VBSM_Dataset_Condensation \
    --fkd-path ../relabel/FKD_cutmix_fp16FKD_IPC_50  \

# CUDA_VISBLE_DEVICES=0 python train_FKD.py \
#     --wandb-project 'final_rn18_fkd' \
#     --batch-size 64 \
#     --model "ResNet18" \
#     --cos --loss-type "mse_gt" --ce-weight 0.15 \
#     -j 4 --gradient-accumulation-steps 1 \
#     -T 20 --sgd --sgd-lr 0.1 --adamw-lr 0.001 \
#     --mix-type 'cutmix' \
#     --output-dir ./save/final_cw128_fkd/ \
#     --train-dir ../../recover_Data/GVBSM_CIFAR_100_Recover_IPC_1_1 \
#     --val-dir /data4t2/ZHOU/G_VBSM_Dataset_Condensation \
#     --fkd-path ../relabel/FKD_cutmix_fp16FKD_IPC_1  # model in [ConvNetW128, ResNet18]
