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
import watermarks

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
parser.add_argument('--attack_method', default='none', help='attack method')
parser.add_argument('--backdoor_type', default='none', help='backdoor method')
parser.add_argument('--pruning_rates', nargs='+', default=[0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.72, 0.74, 0.76, 
                                                           0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94], 
                    type=float, help='percentages (list) of how many weights to prune')
parser.add_argument('--epochs', default=2, type=int, help='number of epochs attacked')
parser.add_argument('--loss_lambda', default=0.5, type=float, help='used in backdoor and random-smoothing')
parser.add_argument('--trg_set_size', default=500, type=int, help='the batch size')
parser.add_argument('--wm_batch_size', default=16, type=int, help='the batch size')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lradj', default=0.1, type=int, help='multiple the lr by lradj every 20 epochs')
parser.add_argument('--optim', default='SGD', help='optimizer (default SGD)')
parser.add_argument('--sched', default='MultiStepLR', help='scheduler (default MultiStepLR)')
parser.add_argument('--fpmethod', default='none', help='fingerprint method')
parser.add_argument('--D1_batch_size', default=64, type=int, help='the batch size')
parser.add_argument('--D2_batch_size', default=16, type=int, help='the batch size')
parser.add_argument('--D1_number', default=20, type=int, help='the number of D1 attacker used')
parser.add_argument('--D2_number', default=20, type=int, help='the number of D2 attacker used')
parser.add_argument('--loadmodel', default='', help='path which model(f0) should be load')
parser.add_argument('--test_type', default='test_B', help='test type')

# cuda
parser.add_argument('--cuda', default='cuda:0', help='set cuda (e.g. cuda:0)')

args = parser.parse_args()

try:
    device = torch.device(args.cuda) if torch.cuda.is_available() else 'cpu'

    cwd = os.getcwd()

    if args.attack_method == 'backdoor':
        log_dir = os.path.join(cwd, 'log', str(args.dataset), str(args.arch), str(args.test_type), str(args.fpmethod), str(args.attack_method) + '_' + str(args.backdoor_type))
    else:
        log_dir = os.path.join(cwd, 'log', str(args.dataset), str(args.arch), str(args.test_type), str(args.fpmethod), str(args.attack_method))
    os.makedirs(log_dir, exist_ok=True)
    configfile = os.path.join(log_dir, time.strftime("%Y%m%d-%H%M%S_") + 'conf_' + str(args.runname) + '.txt')
    logfile = os.path.join(log_dir, time.strftime("%Y%m%d-%H%M%S_") + 'log_' + str(args.runname) + '.txt')
    set_up_logger(logfile)

    with open(configfile, 'w') as f:
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))
    if args.attack_method == 'backdoor':
        result_dir = os.path.join(cwd, 'result', str(args.test_type), str(args.dataset) + '_' + str(args.arch),  str(args.fpmethod), str(args.attack_method) + '_' + str(args.backdoor_type))
    else:
        result_dir = os.path.join(cwd, 'result', str(args.test_type), str(args.dataset) + '_' + str(args.arch),  str(args.fpmethod), str(args.attack_method))
    os.makedirs(result_dir, exist_ok=True)
    resultfile = os.path.join(result_dir, str(args.runname) + '.txt')

    train_db_path = os.path.join(cwd, 'data')
    test_db_path = os.path.join(cwd, 'data')
    transform_train, transform_test = get_data_transforms(args.dataset)
    D1_train_set, D1_test_set, valid_set = get_dataset(args.dataset, train_db_path, test_db_path, transform_train, transform_test, size_train=args.D1_number, size_test=200)
    D1_train_loader = torch.utils.data.DataLoader(D1_train_set, batch_size=args.D1_batch_size, shuffle=False, drop_last=True)
    D1_test_loader = torch.utils.data.DataLoader(D1_test_set, batch_size=args.D1_batch_size, shuffle=False, drop_last=True)
    logging.info('Size of D1 training set: %d, size of D1 testing set: %d' % (len(D1_train_set), len(D1_test_set)))

    D2_targets = torch.tensor(np.loadtxt(os.path.join(os.path.join(cwd, 'data', 'fingerprint_set', args.dataset, args.arch, args.fpmethod, args.loadmodel), 'labels.txt'))).long()
    D2_train_set = []
    D2_test_set = []
    for i in range(0, int(args.D2_number)):
        D2_train_set.append((torch.load(os.path.join(os.path.join(cwd, 'data', 'fingerprint_set', args.dataset, args.arch, args.fpmethod, args.loadmodel), 
                        'pics', f'{str(i+1)}.pt'), map_location=device), D2_targets[i]))
    for i in range(500, 700):
        D2_test_set.append((torch.load(os.path.join(os.path.join(cwd, 'data', 'fingerprint_set', args.dataset, args.arch, args.fpmethod, args.loadmodel), 
                        'pics', f'{str(i+1)}.pt'), map_location=device), D2_targets[i]))
    D2_train_loader = torch.utils.data.DataLoader(D2_train_set, batch_size=args.D2_batch_size, shuffle=False, drop_last=True)
    D2_test_loader = torch.utils.data.DataLoader(D2_test_set, batch_size=args.D2_batch_size, shuffle=False, drop_last=True)
    logging.info('Size of D2 training set: %d, size of D2 testing set: %d' % (len(D2_train_set), len(D2_test_set)))
    
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
    E1 = test(net, criterion, D1_test_loader, device, net_0)
    logging.info("Test acc: %.3f%%" % test_acc)
    
    logging.info("Initial test D2 dataset.")
    test_acc = test(net, criterion, D2_test_loader, device)
    E2 = test(net, criterion, D2_test_loader, device, net_0)
    logging.info("Test acc: %.3f%%" % test_acc)
    logging.info("E1: %.3f%% E2: %.3f%%" % (E1, E2))
    with open(resultfile, 'a') as f:
        f.write('%.3f%% \t%.3f%%\n' % (100. - E2, 100. - E1))

    optimizer, scheduler = set_up_optim_sched(args, net)

    if args.attack_method == 'backdoor':
        logging.info("backdoor attack start......")
        transform = get_wm_transform(args.dataset)
        if args.dataset == 'caltech-101':
            trigger_set = get_trg_set(os.path.join(cwd, 'data', 'trigger_set', args.dataset, args.arch, args.backdoor_type, 'caltech101_' + args.arch + '_' + args.backdoor_type), 
                                  'labels.txt', args.trg_set_size, transform)
        else:
            trigger_set = get_trg_set(os.path.join(cwd, 'data', 'trigger_set', args.dataset, args.arch, args.backdoor_type, args.dataset + '_' + args.arch + '_' + args.backdoor_type), 
                                  'labels.txt', args.trg_set_size, transform)
        wm_loader = torch.utils.data.DataLoader(trigger_set, batch_size=args.wm_batch_size, shuffle=False)
        logging.info('Size of backdoor set: %d ' % len(trigger_set))
        for epoch in range(args.epochs):
            print('\nEpoch: %d' % epoch)
            net.train()

            train_losses = []
            train_loss = 0
            E1, E2 = 0, 0
            D1_correct, wm_correct = 0, 0
            D1_total, wm_total = 0, 0

            for batch_idx, [(D1_inputs, D1_targets), (wm_inputs, wm_targets)] in enumerate(zip(D1_train_loader, cycle(wm_loader))):
                
                print('\nBatch: %d' % batch_idx)

                D1_inputs, D1_targets = D1_inputs.to(device), D1_targets.to(device)
                wm_inputs, wm_targets = wm_inputs.to(device), wm_targets.to(device)

                net_0.eval()
                D1_outputs_0 = net_0(D1_inputs)
                _, D1_predicted_0 = torch.max(D1_outputs_0.data, 1)

                optimizer.zero_grad()
                D1_outputs = net(D1_inputs)
                D1_loss = criterion(D1_outputs, D1_predicted_0)

                wm_outputs = net(wm_inputs)
                wm_loss = criterion(wm_outputs, wm_targets)

                loss = D1_loss * args.loss_lambda + wm_loss
                loss.backward(retain_graph=True)

                optimizer.step()

                train_losses.append(loss.item())

                _, D1_predicted = torch.max(D1_outputs.data, 1)
                D1_total += D1_targets.size(0)
                D1_correct += D1_predicted.eq(D1_predicted_0.data).cpu().sum()
                E1 = 100. * D1_correct / D1_total

                _, wm_predicted = torch.max(wm_outputs.data, 1)
                wm_total += wm_targets.size(0)
                wm_correct += wm_predicted.eq(wm_targets.data).cpu().sum()
                wm_acc = 100. * wm_correct / wm_total

                progress_bar(batch_idx, len(D1_train_loader), 'Loss: %.3f | E1: %.3f%% (%d/%d) | WM_acc: %.3f%% (%d/%d)' %
                        (np.average(train_losses), E1, D1_correct, D1_total, wm_acc, wm_correct, wm_total))

            train_loss = np.average(train_losses)

            logging.info(('Epoch %d: Train loss: %.3f | E1: %.3f%% (%d/%d) | WM_acc: %.3f%% (%d/%d)'
                    % (epoch, train_loss, E1, D1_correct, D1_total, wm_acc, wm_correct, wm_total)))

            logging.info("Test E1 on D1 dataset.")
            E1 = test(net, criterion, D1_test_loader, device, net_0)
            logging.info("Test E2 on D2 dataset.")
            E2 = test(net, criterion, D2_test_loader, device, net_0)
            logging.info("E1: %.3f%% E2: %.3f%%" % (E1, E2))
            with open(resultfile, 'a') as f:
                f.write('%.3f%% \t%.3f%%\n' % (100. - E2, 100. - E1))
            scheduler.step()

    elif args.attack_method == 'fine-tune':
        logging.info("fine-tune attack start......")
        for epoch in range(args.epochs):
            print('\nEpoch: %d' % epoch)
            net.train()

            train_losses = []
            train_loss = 0
            D1_correct, D1_total = 0, 0

            for batch_idx, (D1_inputs, D1_targets) in enumerate(D1_train_loader):
                
                print('\nBatch: %d' % batch_idx)

                D1_inputs, D1_targets = D1_inputs.to(device), D1_targets.to(device)
                optimizer.zero_grad()
                D1_outputs = net(D1_inputs)
                loss = criterion(D1_outputs, D1_targets)
                loss.backward(retain_graph=True)
                optimizer.step()
                train_losses.append(loss.item())

                _, D1_predicted = torch.max(D1_outputs.data, 1)
                D1_total += D1_targets.size(0)
                D1_correct += D1_predicted.eq(D1_targets.data).cpu().sum()
                D1_acc = 100. * D1_correct / D1_total

                progress_bar(batch_idx, len(D1_train_loader), 'Loss: %.3f | D1_acc: %.3f%% (%d/%d)' %
                        (np.average(train_losses), D1_acc, D1_correct, D1_total))

            train_loss = np.average(train_losses)

            logging.info(('Epoch %d: Train loss: %.3f | D1_acc: %.3f%% (%d/%d)'
                    % (epoch, train_loss, D1_acc, D1_correct, D1_total)))

            logging.info("Test E1 on D1 dataset.")
            E1 = test(net, criterion, D1_test_loader, device, net_0)
            logging.info("Test E2 on D2 dataset.")
            E2 = test(net, criterion, D2_test_loader, device, net_0)
            logging.info("E1: %.3f%% E2: %.3f%%" % (E1, E2))
            with open(resultfile, 'a') as f:
                f.write('%.3f%% \t%.3f%%\n' % (100. - E2, 100. - E1))
            scheduler.step()

    elif args.attack_method == 'pruning':
        logging.info("pruning attack start......")
        for pruning_rate in args.pruning_rates:
            pruning_rate = float(pruning_rate)
            prune(net, args.arch, pruning_rate)

            logging.info("Test E1 on D1 dataset.")
            E1 = test(net, criterion, D1_test_loader, device, net_0)
            logging.info("Test E2 on D2 dataset.")
            E2 = test(net, criterion, D2_test_loader, device, net_0)
            logging.info("E1: %.3f%% E2: %.3f%%" % (E1, E2))
            with open(resultfile, 'a') as f:
                f.write('%.3f%% \t%.3f%%\n' % (100. - E2, 100. - E1))
    
    elif args.attack_method == 'random-smoothing':
        logging.info("random-smoothing attack start......")

        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for noise_lambda in [0.1 ,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]:
                logging.info("Test E1 on D1 dataset.")
                for batch_idx, (inputs, targets) in enumerate(D1_test_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    _, targets = torch.max(outputs.data, 1)
                    for i in range(10):
                        inputs_random = inputs + noise_lambda * torch.randn_like(inputs)
                        outputs_random = net(inputs_random)
                        _, predicted_random = torch.max(outputs_random.data, 1)
                        if i == 0:
                            predicted = predicted_random.unsqueeze(0)
                        else:
                            predicted = torch.cat((predicted, predicted_random.unsqueeze(0)), 0)
                    predicted, _ = torch.mode(predicted, dim=0)
                    total += targets.size(0)
                    correct += predicted.eq(targets.data).cpu().sum()
                E1 = 100. * correct / total

                logging.info("Test E2 on D2 dataset.")
                for batch_idx, (inputs, targets) in enumerate(D2_test_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    _, targets = torch.max(outputs.data, 1)
                    for i in range(20):
                        inputs_random = inputs + noise_lambda * torch.randn_like(inputs)
                        outputs_random = net(inputs_random)
                        _, predicted_random = torch.max(outputs_random.data, 1)
                        if i == 0:
                            predicted = predicted_random.unsqueeze(0)
                        else:
                            predicted = torch.cat((predicted, predicted_random.unsqueeze(0)), 0)
                    predicted, _ = torch.mode(predicted, dim=0)
                    total += targets.size(0)
                    correct += predicted.eq(targets.data).cpu().sum()
                E2 = 100. * correct / total

                logging.info("E1: %.3f%% E2: %.3f%%" % (E1, E2))
                with open(resultfile, 'a') as f:
                    f.write('%.3f%% \t%.3f%%\n' % (100. - E2, 100. - E1))
    
except Exception as e:
    msg = 'An error occurred during training in ' + args.runname + ': ' + str(e)
    logging.error(msg)

    traceback.print_tb(e.__traceback__)
