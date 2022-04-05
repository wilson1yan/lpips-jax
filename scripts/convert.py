import numpy as np
import torch
import pickle
import os.path as osp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--net_type', type=str, default='alexnet')
parser.add_argument('-n', '--net_path', type=str, default='weights/alexnet_torch.pth',
                    help='Path to pretrained net checkpoint')
parser.add_argument('-l', '--lin_path', type=str, default='weights/alexnet_linear_torch.pth',
                   help='Can be None. Path to pretrained LPIPs linear checkpoint')
args = parser.parse_args()


sd = torch.load(args.net_path, map_location='cpu')
sd = {k: v.numpy() for k, v in sd.items()}

def Conv(prefix):
    return {
        'kernel': np.transpose(sd[f'{prefix}.weight'], (2, 3, 1, 0)),
        'bias': sd[f'{prefix}.bias']
    }


if args.net_type == 'alexnet':
    params = {
        'AlexNet_0': {
            'Conv_0': Conv('features.0'),
            'Conv_1': Conv('features.3'),
            'Conv_2': Conv('features.6'),
            'Conv_3': Conv('features.8'),
            'Conv_4': Conv('features.10')
        }
    }
elif args.net_type in ['vgg16']: 
    layers = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
    params = {}
    for i, lyr in enumerate(layers):
        params[f'Conv_{i}'] = Conv(f'features.{lyr}')
    
    params = {'VGG16_0': params}
else:
    raise ValueError(f'Unsupported net_type:', args.net_type)

if args.lin_path is not None:
    sd = torch.load(args.lin_path, map_location='cpu')
    sd = {k: v.numpy() for k, v in sd.items()}
    for i in range(5):
        params[f'NetLinLayer_{i}'] = {
            'kernel': sd[f'lin{i}.model.1.weight'].T
        }

pickle.dump(params, open(osp.join('weights', f'{args.net_type}.ckpt'), 'wb'))