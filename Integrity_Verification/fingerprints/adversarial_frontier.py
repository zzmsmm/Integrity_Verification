from fingerprints.base import FpMethod

import os
import logging
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from helpers.utils import *
from helpers.loaders import *


class AdversarialFrontier(FpMethod):
    def __init__(self, args):
        super().__init__(args)
        self.path = os.path.join(os.getcwd(), 'data', 'fingerprint_set')
        os.makedirs(self.path, exist_ok=True)

    def gen_fingerprints(self, model, criterion, device):
        print('Generating fingerprints. Type = ' + 'adversarial_frontier')

        # for i in range(self.size):
        while True:
            model.eval()

            img = torch.rand(3, 32, 32).to(device)
            img1 = img.unsqueeze(0)
            fp_lbl = torch.argmax(model(img1), dim=1)
            fp_lbl = (fp_lbl + random.randint(1, self.num_classes - 1)) % self.num_classes

            adv = torch.rand(3, 32, 32).to(device)
            adv.requires_grad = True
            optimizer = optim.Adam([adv], lr=0.1)

            for epoch in range(1000):
                optimizer.zero_grad()
                img2 = (img + adv).unsqueeze(0)
                output = model(img2)
                pre_lbl = torch.argmax(model(img2), dim=1)
                if pre_lbl == fp_lbl:
                    self.fingerprint_set.append((img + adv, fp_lbl.clone().detach()))
                    break
                target = fp_lbl.reshape(-1)  # key point
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            if len(self.fingerprint_set) == self.size:
                break

        if self.save_fp:
            save_triggerset(self.fingerprint_set, os.path.join(self.path, self.dataset, self.arch, 'adversarial_frontier'), self.loadmodel)
            print('fingerprints generation done')