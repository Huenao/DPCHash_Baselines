import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from torchvision import models
from models.VTS import CONFIGS, VisionTransformer
from utils.mask_generator import *

class IPHash(nn.Module):
    def __init__(self, bit, temperature, config):
        super(IPHash, self).__init__()
        vts_config = CONFIGS['VTS16']
        vts_config.pretrained_dir = 'checkpoints/imagenet21k+imagenet2012_ViT-B_16-224.npz'
        self.teacher = VisionTransformer(vts_config, 224, num_classes=1000, zero_head=False, vis=True)
        self.vit = VisionTransformer(vts_config, 224, num_classes=1000, zero_head=False, vis=True)
        self.teacher.load_from(np.load(vts_config.pretrained_dir))
        self.vit.load_from(np.load(vts_config.pretrained_dir))

        # [1000, 768]
        self.centers = self.teacher.head.weight.data 
        self.centers = self.centers.to(config["device"])
        self.temperature = temperature
        self.mask_ratio = config["mask_ratio"]
        self.alpha = config["alpha"]
        self.gamma = config["gamma"]
        self.beta = config["beta"]
        self.mask_generator = RandomMaskingGenerator(14, self.mask_ratio)
        
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        self.encoder = nn.Linear(768, bit)
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
            h_centers = self.encoder(self.centers)

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

