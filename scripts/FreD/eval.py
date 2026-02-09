import os
import torch
import numpy as np
import argparse

from utils import (get_dataset,get_network,get_eval_pool,evaluate_synset,set_seed,ParamDiffAug)


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, \
    testloader, loader_train_dict, class_map, class_map_inv \
    = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)

    args.channel = channel
    args.im_size = im_size
    args.num_classes = num_classes
    args.mean = mean
    args.std = std

    if args.dsa:
    # args.epoch_eval_train = 1000
        args.dc_aug_param = None
    args.dsa_param = ParamDiffAug()
    dsa_params = args.dsa_param
    args.dsa_param = dsa_params
    
    # synthetic data from **https://drive.google.com/drive/folders/1r1OMVv9llejGmpHfK5DpW4m57Dz_SZ2n**
    print(f"Loading synthetic data from {args.load_path}")
    image_syn = torch.load(os.path.join(args.load_path,f"FreD_ipc{args.ipc}#images_best.pt"), map_location='cpu')
    label_syn = torch.load(os.path.join(args.load_path,f"FreD_ipc{args.ipc}#labels_best.pt"), map_location='cpu')

    image_syn = image_syn.to(device)
    label_syn = label_syn.to(device)

    print(f"Synthetic images: {image_syn.shape}")
    print(f"Synthetic labels: {label_syn.shape}")

    # evaluate
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    for model_eval in model_eval_pool:
        print("=" * 60)
        print(f"Evaluating model_train = {args.model}, model_eval = {model_eval}")

        if args.dsa:
            print('DSA augmentation strategy: \n', args.dsa_strategy)
            print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
        else:
            print('DC augmentation parameters: \n', args.dc_aug_param)
        
        accs_test = []
        accs_train = []

        for it_eval in range(args.num_eval):
            net = get_network(model_eval, channel, num_classes, im_size).to(device)
            args.lr_net = args.lr_teacher
            _, acc_train, acc_test = evaluate_synset(it_eval, net, image_syn, label_syn, testloader, args)

            accs_train.append(acc_train)
            accs_test.append(acc_test)

        accs_test = np.array(accs_test)
        print(
            f"[{model_eval}] "
            f"Test Acc: mean={accs_test.mean():.4f}, std={accs_test.std():.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--subset', type=str, default='imagenette')
    parser.add_argument('--model', type=str, default='ConvNet')
    parser.add_argument('--eval_mode', type=str, default='S')
    parser.add_argument('--num_eval', type=int, default=5)
    parser.add_argument('--ipc', type=int, required=True)
    
    parser.add_argument('--epoch_eval_train', type=int, default=1000)
    parser.add_argument('--batch_train', type=int, default=256)
    parser.add_argument('--batch_real', type=int, default=256)
    parser.add_argument('--lr_teacher', type=float, default=0.01)

    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--load_path', type=str, required=True)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--zca', action='store_true')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'], help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    
    args = parser.parse_args()
    set_seed(args.seed)

    main(args)
