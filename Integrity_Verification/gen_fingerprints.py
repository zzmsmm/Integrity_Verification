""" generate fingerprints """

import argparse
import time
import traceback

from babel.numbers import format_decimal

# import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.backends import cudnn

import models
import fingerprints

from helpers.utils import *
from helpers.loaders import *
from helpers.image_folder_custom_class import *


# possible models to use
model_names = sorted(name for name in models.__dict__ if name.islower() and callable(models.__dict__[name]))
# print('models : ', model_names)

# possible fingerprinting methods to use
fingerprinting_methods = sorted(
    fingerprint for fingerprint in fingerprints.__dict__ if callable(fingerprints.__dict__[fingerprint]))
# print('fingerprints: ', fingerprinting_methods)

# set up argument parser
parser = argparse.ArgumentParser(description='Extract fingerprints from models.')

# model and dataset
parser.add_argument('--dataset', default='cifar10', help='the dataset to train on [cifar10]')
parser.add_argument('--num_classes', default=10, type=int, help='number of classes for classification')
parser.add_argument('--arch', metavar='ARCH', default='cnn_cifar10', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: cnn_cifar10)')

# fingerprint related
parser.add_argument('--method', default=None, choices=fingerprinting_methods,
                    help='fingerprinting method: ' + ' | '.join(
                        fingerprinting_methods) + ' (default: random)')
parser.add_argument('--save_fp', action='store_true', help='save generated fingerprints?')
parser.add_argument('--fp_set_size', default=20, type=int, help='the size of the fingerprint set (default: 20)')
parser.add_argument('--loadmodel', default='', help='path which model should be load for pretrained embed type')


# hyperparameters
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lradj', default=0.1, type=int, help='multiple the lr by lradj every 20 epochs')
parser.add_argument('--optim', default='SGD', help='optimizer (default SGD)')
parser.add_argument('--sched', default='MultiStepLR', help='scheduler (default MultiStepLR)')
# cuda
parser.add_argument('--cuda', default=None, help='set cuda (e.g. cuda:0)')

args = parser.parse_args()

try:
    device = torch.device(args.cuda) if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    cwd = os.getcwd()


except Exception as e:
    msg = 'An error occured during setup: ' + str(e)

try:
    generation_time = 0

    fp_method = fingerprints.__dict__[args.method](args)

    net = models.__dict__[args.arch](num_classes=args.num_classes)
    net.load_state_dict(torch.load(os.path.join('checkpoint', 'clean', args.loadmodel + '.ckpt')))
    net.to(device)
    start_time = time.time()

    if args.method == 'Random' or args.method == 'Canonical':
        fp_method.gen_fingerprints(net, device)

    elif args.method == 'AdversarialFrontier' or args.method == 'AeFrontier' or args.method == 'AdversarialCenter':
        criterion = nn.CrossEntropyLoss()
        fp_method.gen_fingerprints(net, criterion, device)
    
    generation_time = time.time() - start_time

except Exception as e:
    msg = 'An error occured during fingerprint generation in ' + args.loadmodel + ': ' + str(e)

    traceback.print_tb(e.__traceback__)
