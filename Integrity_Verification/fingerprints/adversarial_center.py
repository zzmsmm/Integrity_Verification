from fingerprints.base import FpMethod

import os
import logging
import random
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from helpers.utils import *
from helpers.loaders import *

#### 待重新实现

class AdversarialCenter(FpMethod):
    def __init__(self, args):
        super().__init__(args)
        self.path = os.path.join(os.getcwd(), 'data', 'fingerprint_set')
        os.makedirs(self.path, exist_ok=True)

    def gen_fingerprints(self, model, criterion, device):
        print('Generating fingerprints. Type = ' + 'adversarial_center')

        datasets_dict = {'cifar10': datasets.CIFAR10, 'fashionmnist': datasets.FashionMNIST, 'mnist': datasets.MNIST}
        _, transform = get_data_transforms(self.dataset)

        if self.dataset == 'caltech-101':
            caltech_directory = os.path.join('./data', 'caltech-101')
            fp_set = datasets.ImageFolder(os.path.join(caltech_directory, 'train'),
                                         transform=transform)
        elif self.dataset == 'tiny-imagenet':
            id_dict = {}
            for i, line in enumerate(open(os.path.join('./data', 'tiny-imagenet-200/wnids.txt'), 'r')):
                id_dict[line.replace('\n', '')] = i
            fp_set = TrainTinyImageNetDataset(id=id_dict, transform=transform)
        else:
            fp_set = datasets_dict[self.dataset](root='./data', train=True, download=True, transform=transform)

        for i in random.sample(range(len(fp_set)), len(fp_set)):  # iterate randomly
            model.eval()

            img, _ = fp_set[i]
            img = img.to(device)
            img1 = img.unsqueeze(0)

            fp_lbl = torch.argmax(model(img1), dim=1)

            adv = torch.rand(3, 32, 32).to(device)
            adv.requires_grad = True
            optimizer = optim.Adam([adv], lr=0.01)

            for epoch in range(1000):
                optimizer.zero_grad()
                img2 = (img + adv).unsqueeze(0)
                output = model(img2)
                pre_lbl = torch.argmax(model(img2), dim=1)
                if pre_lbl == fp_lbl and epoch > 5:
                    self.fingerprint_set.append((img + adv, fp_lbl.clone().detach()))
                    break
                target = fp_lbl.reshape(-1)  # key point
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            if len(self.fingerprint_set) == self.size:
                break

        if self.save_fp:
            save_triggerset(self.fingerprint_set, os.path.join(self.path, self.dataset, self.arch, 'adversarial_center'), self.loadmodel)
            print('fingerprints generation done')