import argparse
import traceback

from babel.numbers import format_decimal

from torch.backends import cudnn

import numpy as np
import models
import torch
import torchvision
import torchvision.transforms as transforms
import copy
from itertools import cycle

from helpers.utils import *
from helpers.loaders import *
from helpers.image_folder_custom_class import *
from trainer import test
from attacks.pruning import prune

# possible models to use
model_names = sorted(name for name in models.__dict__ if name.islower() and callable(models.__dict__[name]))
# print('models : ', model_names)

# set up argument parser
parser = argparse.ArgumentParser(description='Train models without watermarks.')

# model and dataset
parser.add_argument('--dataset', default='cifar10', help='the dataset to train on [cifar10]')
parser.add_argument('--num_classes', default=10, type=int, help='number of classes for classification')
parser.add_argument('--arch', metavar='ARCH', default='cnn_cifar10', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: cnn_cifar10)')

# hyperparameters
parser.add_argument('--runname', default='cifar10_custom_cnn', help='the exp name')
parser.add_argument('--initial_attack_method', default='pruning', help='initial attack method')
parser.add_argument('--pruning_rate', default=0.7, type=float, help='percentages (list) of how many weights to prune')
parser.add_argument('--epochs_initial_attack', default=2, type=int, help='number of epochs initial attacked')
parser.add_argument('--epochs_advft_attack', default=5, type=int, help='number of epochs advft attacked')
parser.add_argument('--loss_lambda', default=0.5, type=float, help='D2_loss')
parser.add_argument('--D1_batch_size', default=64, type=int, help='the batch size')
parser.add_argument('--D2_batch_size', default=16, type=int, help='the batch size')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lradj', default=0.1, type=int, help='multiple the lr by lradj every 20 epochs')
parser.add_argument('--optim', default='SGD', help='optimizer (default SGD)')
parser.add_argument('--sched', default='MultiStepLR', help='scheduler (default MultiStepLR)')
parser.add_argument('--fpmethod', default='none', help='fingerprint method (D2)')
parser.add_argument('--use_watermark', action='store_true', help='D1 use backdoor watermark')
parser.add_argument('--D1_proportion', default=0.1, type=float, help='the proportion of D1 attacker used')
parser.add_argument('--D1_number', default=20, type=int, help='the number of D1 attacker used')
parser.add_argument('--D2_number', default=20, type=int, help='the number of D2 attacker used')
parser.add_argument('--loadmodel', default='', help='path which model(f0) should be load')
parser.add_argument('--test_type', default='test_A', help='test type')

# cuda
parser.add_argument('--cuda', default='cuda:0', help='set cuda (e.g. cuda:0)')

args = parser.parse_args()

try:
    device = torch.device(args.cuda) if torch.cuda.is_available() else 'cpu'
    print(device)

    cwd = os.getcwd()
    if args.test_type == 'test_A':
        log_dir = os.path.join(cwd, 'log', str(args.dataset), str(args.arch), str(args.test_type), str(args.fpmethod), 
                               str(args.D1_number) + '_' + str(args.D2_number))
        result_dir = os.path.join(cwd, 'result', str(args.test_type), str(args.dataset) + '_' + str(args.arch), str(args.fpmethod),
                                str(args.D1_number) + '_' + str(args.D2_number))
    elif args.test_type == 'test_B':
        pass
    elif args.test_type == 'test_C':
        log_dir = os.path.join(cwd, 'log', str(args.dataset), str(args.arch), str(args.test_type), str(args.fpmethod))
        result_dir = os.path.join(cwd, 'result', str(args.test_type), str(args.dataset) + '_' + str(args.arch), str(args.fpmethod))
    
    os.makedirs(log_dir, exist_ok=True)
    configfile = os.path.join(log_dir, time.strftime("%Y%m%d-%H%M%S_") + 'conf_' + str(args.runname) + '.txt')
    logfile = os.path.join(log_dir, time.strftime("%Y%m%d-%H%M%S_") + 'log_' + str(args.runname) + '.txt')
    set_up_logger(logfile)

    with open(configfile, 'w') as f:
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))

    os.makedirs(result_dir, exist_ok=True)
    resultfile = os.path.join(result_dir, str(args.runname) + '.txt')

    if args.use_watermark:
        if args.fpmethod == 'canonical':
            D1_train_set = []
            D1_test_set = []
            D1_targets = torch.tensor(np.loadtxt(os.path.join(os.path.join(cwd, 'data', 'trigger_set', args.dataset, args.arch, 'content', args.loadmodel), 'labels.txt'))).long()
            for i in range(0, int(args.D1_number)):
                D1_train_set.append((torch.load(os.path.join(os.path.join(cwd, 'data', 'trigger_set', args.dataset, args.arch, 'content', args.loadmodel), 
                                'pics', f'{str(i+1)}.pt'), map_location=device), D1_targets[i]))
            for i in range(500, 700):
                D1_test_set.append((torch.load(os.path.join(os.path.join(cwd, 'data', 'trigger_set', args.dataset, args.arch, 'content', args.loadmodel), 
                                'pics', f'{str(i+1)}.pt'), map_location=device), D1_targets[i]))
        else:
            D1_transform = get_wm_transform(args.dataset)
            D1_train_set, D1_test_set = get_trg_set_split(os.path.join(cwd, 'data', 'trigger_set', args.dataset, args.arch, 'content', args.loadmodel), 'labels.txt', args.D1_number, D1_transform)
    else:
        train_db_path = os.path.join(cwd, 'data')
        test_db_path = os.path.join(cwd, 'data')
        transform_train, transform_test = get_data_transforms(args.dataset)
        D1_train_set, D1_test_set, valid_set = get_dataset(args.dataset, train_db_path, test_db_path, transform_train, transform_test, size_train=args.D1_number, size_test=200)
    D1_train_loader = torch.utils.data.DataLoader(D1_train_set, batch_size=args.D1_batch_size, shuffle=False, drop_last=True)
    D1_test_loader = torch.utils.data.DataLoader(D1_test_set, batch_size=args.D1_batch_size, shuffle=False, drop_last=True)
    logging.info('Size of D1 training set: %d, size of D1 testing set: %d' % (len(D1_train_set), len(D1_test_set)))
    
    D2_train_set = []
    D2_test_set = []
    if args.fpmethod == 'canonical':
        D2_targets = torch.tensor(np.loadtxt(os.path.join(os.path.join(cwd, 'data', 'trigger_set', args.dataset, args.arch, 'content', args.loadmodel), 'labels.txt'))).long()
        for i in range(700, 700 + int(args.D2_number)):
            D2_train_set.append((torch.load(os.path.join(os.path.join(cwd, 'data', 'trigger_set', args.dataset, args.arch, 'content', args.loadmodel), 
                            'pics', f'{str(i+1)}.pt'), map_location=device), D2_targets[i]))
        for i in range(1200, 1400):
            D2_test_set.append((torch.load(os.path.join(os.path.join(cwd, 'data', 'trigger_set', args.dataset, args.arch, 'content', args.loadmodel), 
                            'pics', f'{str(i+1)}.pt'), map_location=device), D2_targets[i]))
    else:
        D2_targets = torch.tensor(np.loadtxt(os.path.join(os.path.join(cwd, 'data', 'fingerprint_set', args.dataset, args.arch, args.fpmethod, args.loadmodel), 'labels.txt'))).long()
        for i in range(0, int(args.D2_number)):
            D2_train_set.append((torch.load(os.path.join(os.path.join(cwd, 'data', 'fingerprint_set', args.dataset, args.arch, args.fpmethod, args.loadmodel), 
                            'pics', f'{str(i+1)}.pt'), map_location=device), D2_targets[i]))
        for i in range(500, 700):
            D2_test_set.append((torch.load(os.path.join(os.path.join(cwd, 'data', 'fingerprint_set', args.dataset, args.arch, args.fpmethod, args.loadmodel), 
                            'pics', f'{str(i+1)}.pt'), map_location=device), D2_targets[i]))
    D2_train_loader = torch.utils.data.DataLoader(D2_train_set, batch_size=args.D2_batch_size, shuffle=False, drop_last=True)
    D2_test_loader = torch.utils.data.DataLoader(D2_test_set, batch_size=args.D2_batch_size, shuffle=False, drop_last=True)
    logging.info('Size of D2 training set: %d, size of D2 testing set: %d' % (len(D2_train_set), len(D2_test_set)))
    
    # set up loss
    criterion = nn.CrossEntropyLoss()

except Exception as e:
    msg = 'An error occurred during setup: ' + str(e)
    logging.error(msg)


try:
    net = models.__dict__[args.arch](num_classes=args.num_classes)
    net.load_state_dict(torch.load(os.path.join('checkpoint', 'clean', args.loadmodel + '.ckpt')))
    net.to(device)
    
    net_0 = copy.deepcopy(net)

    logging.info("Initial test D1 dataset.")
    test_acc = test(net, criterion, D1_test_loader, device)
    logging.info("Test acc: %.3f%%" % test_acc)
    
    logging.info("Initial test D2 dataset.")
    test_acc = test(net, criterion, D2_test_loader, device)
    logging.info("Test acc: %.3f%%" % test_acc)

    if args.initial_attack_method == 'pruning':
        args.pruning_rate = float(args.pruning_rate)
        prune(net, args.arch, args.pruning_rate)

        logging.info("Test D1 dataset after initial attack.")
        test_acc = test(net, criterion, D1_test_loader, device)
        logging.info("Test acc: %.3f%%" % test_acc)

        logging.info("Test D2 dataset after initial attack.")
        test_acc = test(net, criterion, D2_test_loader, device)
        logging.info("Test acc: %.3f%%" % test_acc)
    
    logging.info("Test E1 on D1 dataset.")
    E1 = test(net, criterion, D1_test_loader, device, net_0)
    logging.info("Test E2 on D2 dataset.")
    E2 = test(net, criterion, D2_test_loader, device, net_0)
    logging.info("E1: %.3f%% E2: %.3f%%" % (E1, E2))
    with open(resultfile, 'a') as f:
        f.write('%.3f%% \t%.3f%%\n' % (100. - E2, 100. - E1))

    logging.info("Training Start......")
    optimizer, scheduler = set_up_optim_sched(args, net)

    for epoch  in range(args.epochs_advft_attack):
        print('\nEpoch: %d' % epoch)
        net.train()

        train_losses = []
        train_loss = 0
        E1, E2 = 0, 0
        D1_correct, D2_correct = 0, 0
        D1_total, D2_total = 0, 0

        for batch_idx, [(D1_inputs, D1_targets), (D2_inputs, D2_targets)] in enumerate(zip(D1_train_loader, cycle(D2_train_loader))):
            
            print('\nBatch: %d' % batch_idx)

            D1_inputs, D1_targets = D1_inputs.to(device), torch.randint(0, args.num_classes, (args.D1_batch_size,)).to(device)
            D2_inputs, D2_targets = D2_inputs.to(device), D2_targets.to(device)

            net_0.eval()
            D1_outputs_0 = net_0(D1_inputs)
            D2_outputs_0 = net_0(D2_inputs)
            _, D1_predicted_0 = torch.max(D1_outputs_0.data, 1)
            _, D2_predicted_0 = torch.max(D2_outputs_0.data, 1)

            optimizer.zero_grad()
            D1_outputs = net(D1_inputs)
            D1_loss = criterion(D1_outputs, D1_targets)

            D2_outputs = net(D2_inputs)
            D2_loss = criterion(D2_outputs, D2_predicted_0)

            loss = D1_loss + D2_loss * args.loss_lambda
            loss.backward(retain_graph=True)

            optimizer.step()

            train_losses.append(loss.item())

            _, D1_predicted = torch.max(D1_outputs.data, 1)
            D1_total += D1_targets.size(0)
            D1_correct += D1_predicted.eq(D1_predicted_0.data).cpu().sum()
            E1 = 100. * D1_correct / D1_total

            _, D2_predicted = torch.max(D2_outputs.data, 1)
            D2_total += D2_targets.size(0)
            D2_correct += D2_predicted.eq(D2_predicted_0.data).cpu().sum()
            E2 = 100. * D2_correct / D2_total

            progress_bar(batch_idx, len(D1_train_loader), 'Loss: %.3f | E1: %.3f%% (%d/%d) | E2: %.3f%% (%d/%d)' %
                    (np.average(train_losses), E1, D1_correct, D1_total, E2, D2_correct, D2_total))

        train_loss = np.average(train_losses)

        logging.info(('Epoch %d: Train loss: %.3f | E1: %.3f%% (%d/%d) | E2: %.3f%% (%d/%d)'
                  % (epoch, train_loss, E1, D1_correct, D1_total, E2, D2_correct, D2_total)))

        logging.info("Test E1 on D1 dataset.")
        E1 = test(net, criterion, D1_test_loader, device, net_0)
        logging.info("Test E2 on D2 dataset.")
        E2 = test(net, criterion, D2_test_loader, device, net_0)
        logging.info("E1: %.3f%% E2: %.3f%%" % (E1, E2))
        with open(resultfile, 'a') as f:
            f.write('%.3f%% \t%.3f%%\n' % (100. - E2, 100. - E1))
        scheduler.step()      
        
except Exception as e:
    msg = 'An error occurred during training in ' + args.runname + ': ' + str(e)
    logging.error(msg)

    traceback.print_tb(e.__traceback__)
