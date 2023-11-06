"""Adversarial Frontier Stitching (Merrer et al., 2019)

- needs pretrained model
- generating adversarial inputs with fast gradient sign method

Implementation based on: https://github.com/dunky11/adversarial-frontier-stitching/ (original in tensorflow)"""
from helpers.loaders import get_wm_transform
from watermarks.base import WmMethod

import os
import logging

import torch

from helpers.utils import find_tolerance, fast_gradient_sign, save_triggerset, get_trg_set
from trainer import test, train, train_wo_wms, train_on_augmented
from torchvision.utils import save_image

class FrontierStitching(WmMethod):
    def __init__(self, args):
        super().__init__(args)

        self.path = os.path.join(os.getcwd(), 'data', 'trigger_set')
        os.makedirs(self.path, exist_ok=True)  # path where to save trigger set if has to be generated

    def gen_watermarks(self, model, criterion, device, train_loader, eps):
        true_advs = list()
        false_advs = list()
        max_true_advs = max_false_advs = self.size // 2

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # generate adversary
            x_advs = fast_gradient_sign(x, y, model, criterion, device, eps)
            
            y_preds = torch.argmax(model(x), dim=1)
            y_pred_advs = torch.argmax(model(x_advs), dim=1)

            for x_adv, y_pred_adv, y_pred, y_true in zip(x_advs, y_pred_advs, y_preds, y):
                # x_adv is a true adversary
                if y_pred == y_true and y_pred_adv != y_true and len(true_advs) < max_true_advs:
                    true_advs.append((x_adv, y_true))

                # x_adv is a false adversary
                if y_pred == y_true and y_pred_adv == y_true and len(false_advs) < max_false_advs:
                    false_advs.append((x_adv, y_true))

                if len(true_advs) == max_true_advs and len(false_advs) == max_false_advs:
                    break

        if self.save_wm:
            save_triggerset(true_advs + false_advs, os.path.join(self.path, self.dataset, self.arch, 'frontier_stitching'), self.runname)
            print('watermarks generation done')

        self.trigger_set = true_advs + false_advs

    def embed(self, net, criterion, optimizer, scheduler, train_set, test_set, train_loader, test_loader, valid_loader, device, save_dir):

        logging.info("Embedding watermarks.")

        transform = get_wm_transform(self.dataset)
        self.trigger_set = get_trg_set(os.path.join(self.path, self.dataset, self.arch, 'frontier_stitching', self.runname), 'labels.txt',
                                       self.size, transform)

        self.loader()

        if self.embed_type == 'pretrained':
            # load model
            logging.info("Load model: " + self.loadmodel + ".ckpt")
            net.load_state_dict(torch.load(os.path.join('checkpoint', 'clean', self.loadmodel + '.ckpt')))

            real_acc, wm_acc, val_loss, epoch = train_on_augmented(self.epochs_w_wm, device, net, optimizer, criterion,
                                                               scheduler, self.patience, train_loader, test_loader,
                                                               valid_loader, self.wm_loader, save_dir, self.save_model)

        logging.info("Done embedding.")

        return real_acc, wm_acc, val_loss, epoch