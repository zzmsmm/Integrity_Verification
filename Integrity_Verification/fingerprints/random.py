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


class Random(FpMethod):
    def __init__(self, args):
        super().__init__(args)
        self.path = os.path.join(os.getcwd(), 'data', 'fingerprint_set')
        os.makedirs(self.path, exist_ok=True)

    def gen_fingerprints(self, model, device):
        print('Generating fingerprints. Type = ' + 'random')
        model.eval() # 至关重要
        for i in range(self.size):
            img = torch.rand(3, 32, 32)
            img1 = img.unsqueeze(0)
            img1 = img1.to(device)
            fp_lbl = torch.argmax(model(img1), dim=1)
            # self.fingerprint_set.append((img, torch.tensor(fp_lbl)))
            self.fingerprint_set.append((img, fp_lbl.clone().detach()))
        
        if self.save_fp:
            save_triggerset(self.fingerprint_set, os.path.join(self.path, self.dataset, self.arch, 'random'), self.loadmodel)
            print('fingerprints generation done')
