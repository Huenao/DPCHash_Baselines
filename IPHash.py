import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import IPHash_get_config
from model.data import my_dataloader
from model.VTS import CONFIGS, VisionTransformer
from utils.mask_generator import *
from utils.utils import evalModel, save_config

torch.multiprocessing.set_sharing_strategy('file_system')


class IPHash(nn.Module):
    def __init__(self, bit, config):
        super(IPHash, self).__init__()
        vts_config = CONFIGS[config['backbone']]
        vts_config.pretrained_dir = config['pretrained_dir']

        self.teacher = VisionTransformer(vts_config, 224, num_classes=1000, zero_head=False, vis=True)
        self.vit = VisionTransformer(vts_config, 224, num_classes=1000, zero_head=False, vis=True)
        self.teacher.load_from(np.load(vts_config.pretrained_dir))
        self.vit.load_from(np.load(vts_config.pretrained_dir))

        # [1000, 768]
        self.centers = self.teacher.head.weight.data
        self.temperature = config["temperature"]
        self.mask_ratio = config["mask_ratio"]
        self.alpha = config["alpha"]
        self.gamma = config["gamma"]
        self.beta = config["beta"]
        self.mask_generator = RandomMaskingGenerator(14, self.mask_ratio)
        
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        self.encoder = nn.Linear(vts_config.hidden_size, bit)
        self.hash_norm = nn.BatchNorm1d(bit)


    class Hash(torch.autograd.Function):
        @staticmethod
        def forward(_, input):
            return input.sign()
        @staticmethod
        def backward(_, grad_output):
            return grad_output

    def sim_loss(self, hash_code, features):
        hash_code = F.normalize(hash_code)
        features = F.normalize(features)
        
        cos_hash = hash_code @ hash_code.t()
        cos_feat = features @ features.t()

        weight = torch.tril(torch.ones_like(cos_hash), diagonal=-1).t()
        weight = weight / weight.sum()
        weight = weight.to(cos_hash.device)

        loss = (torch.pow(cos_hash - cos_feat, 2) * weight).sum()
        return loss  

    def get_mask_random(self, batch_size):
        mask = np.stack([self.mask_generator() for i in range(batch_size)])
        mask = torch.from_numpy(mask).float()
        return mask

    def forward(self, x):
        if not self.training:
            features, _ = self.vit(x)
            h = self.encoder(features)
            return h
        else:
            features, gt_logits = self.teacher(x)

            mask = self.get_mask_random(x.shape[0])
            mask_features, _ = self.vit(x, mask)

            h = self.encoder(mask_features)
            h_centers = self.encoder(self.centers.to(h.device))

            b = IPHash.Hash.apply(h)

            h_logits = h @ h_centers.t()

            # control temperature
            h_logits = h_logits / self.temperature
            gt_logits = gt_logits / self.temperature

            gt_logits_soft = torch.softmax(gt_logits, dim=-1)
            h_logits_soft = F.log_softmax(h_logits, dim=-1)
            loss1 = F.kl_div(h_logits_soft, gt_logits_soft, reduction='batchmean')

            loss2 = F.mse_loss(mask_features, features)
            loss3 = self.sim_loss(b, features)
            loss4 = (h.abs() - 1).pow(3).abs().mean()

            loss = self.beta * loss1 + self.gamma * loss2 + loss3 + self.alpha * loss4
            return {'loss': loss, 'KD_loss': loss1, 'RM_loss': loss2, 'GH_loss': loss3, 'Quan_loss': loss4}

        
def trainer(config, bit):
    Best_mAP = 0
    
    train_logfile = open(os.path.join(config['logs_path'], 'train_log.txt'), 'a')
    train_logfile.write(f"***** {config['info']} - {config['backbone']} - {bit}bit *****\n\n")

    """DataLoader"""
    train_loader, test_loader, database_loader, num_train, num_test, num_database = my_dataloader(config)
    
    """Model"""
    device = torch.device('cuda')
    net = IPHash(bit, config)
    net = net.to(device)

    """Optimizer Setting"""
    optimizer = config["optimizer"]["type"]([{"params": net.encoder.parameters(), "lr": config["optimizer"]["lr"]},
                                             {"params": net.vit.parameters(), "lr": config["optimizer"]["backbone_lr"]},
                                            ])

    """Data Parallel"""
    net = torch.nn.DataParallel(net)

    for epoch in range(config["epoch"]):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("%s-%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], config["backbone"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()
        train_loss = 0
        for image, _, _ in train_loader:
            image = image.to(device)

            optimizer.zero_grad()

            losses = net(image)

            loss = losses['loss']
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)
        print("\b\b\b\b\b\b\b loss:%.5f" % train_loss)

        if (epoch + 1) % config["test_map"] == 0:
            net.eval()
            with torch.no_grad():
                Best_mAP = evalModel(test_loader, database_loader, net, Best_mAP, bit, config, epoch, train_logfile)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    start_time = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))
    setup_seed(2022)
    
    config = IPHash_get_config(start_time)
    save_config(config, config["logs_path"])
    
    for bit in config["bit_list"]:
        trainer(config, bit)

