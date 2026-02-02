import torch
import argparse
import numpy as np
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, ParamDiffAug

def main(args):
    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.dsa_param = ParamDiffAug()
    
    args.dc_aug_param = None
    
    channel, im_size, num_classes, class_names, mean, std, _, _, testloader, \
    _, _, _ = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)

    # 2.synthetic data from **https://georgecazenavette.github.io/mtt-distillation/tensors/index.html#tensors **
    print(f"Loading synthetic data from {args.load_path}")
    image_syn = torch.load(f"{args.load_path}/images_best.pt").to(args.device)
    label_syn = torch.load(f"{args.load_path}/labels_best.pt").to(args.device)
    
    image_syn = image_syn.detach()
    label_syn = label_syn.detach()

    print(f"Synthetic Data Shape: {image_syn.shape}")

    # 3. evaluate
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    
    results = {}

    for model_name in model_eval_pool:

        if args.dsa:
            print('DSA augmentation strategy: \n', args.dsa_strategy)
            print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
        else:
            print('DC augmentation parameters: \n', args.dc_aug_param)
                    
        print(f"\n{'='*20} Evaluating {model_name} {'='*20}")
        accs_test = []

        for it_eval in range(args.num_eval):
           
            net_eval = get_network(model_name, channel, num_classes, im_size).to(args.device)
            
            _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn, label_syn, testloader, args, texture=args.texture)
            
            accs_test.append(acc_test)
            print(f"Run {it_eval+1}/{args.num_eval}: Test Acc = {acc_test :.3f}")

        accs_test = np.array(accs_test)
        results[model_name] = (np.mean(accs_test), np.std(accs_test))
        print(f"\nFinal Result for {model_name}: Mean = {np.mean(accs_test):.3f} std = {np.std(accs_test):.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MTT Evaluation Script')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--subset', type=str, default='imagenette')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--load_path', type=str, default='../downloads/cifar10/', help='Path to best.pt files')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')
    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--lr_net', type=float, default=0.01, help='Learning rate for model training')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')

    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',help='differentiable Siamese augmentation strategy')
    
    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")
    parser.add_argument('--texture', action='store_true', help="will distill textures instead")

    args = parser.parse_args()
    main(args)
