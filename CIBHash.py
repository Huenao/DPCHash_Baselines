import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import CIBHash_get_config
from model.data import CIB_CIFAR_DataLoader, get_data_CIB
from model.vit import CONFIGS, VisionTransformer
from utils.utils import evalModel, save_config

torch.multiprocessing.set_sharing_strategy('file_system')

class NtXentLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NtXentLoss, self).__init__()
        #self.batch_size = batch_size
        self.temperature = temperature
        #self.device = device

        #self.mask = self.mask_correlated_samples(batch_size)
        self.similarityF = nn.CosineSimilarity(dim = 2)
        self.criterion = nn.CrossEntropyLoss(reduction = 'sum')
    

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size 
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask
    

    def forward(self, z_i, z_j, device):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        batch_size = z_i.shape[0]
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarityF(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        #sim = 0.5 * (z_i.shape[1] - torch.tensordot(z.unsqueeze(1), z.T.unsqueeze(0), dims = 2)) / z_i.shape[1] / self.temperature

        sim_i_j = torch.diag(sim, batch_size )

        sim_j_i = torch.diag(sim, -batch_size )
        
        mask = self.mask_correlated_samples(batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).view(N, 1)
        negative_samples = sim[mask].view(N, -1)

        labels = torch.zeros(N).to(device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

class CIBHash(nn.Module):
    def __init__(self, bit, config):
        super(CIBHash, self).__init__()
        vit_config = CONFIGS[config['backbone']]
        vit_config.pretrained_dir = config['pretrained_dir']
        self.vit = VisionTransformer(vit_config, 224, num_classes=1000, zero_head=False, vis=True)
        self.vit.load_from(np.load(vit_config.pretrained_dir))
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.eval()

        # self.vgg = models.vgg16(pretrained=True)
        # self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:6])
        # for param in self.vgg.parameters():
        #     param.requires_grad = False

        self.kl_weight = config['weight']

        self.encoder = nn.Linear(vit_config.hidden_size, bit)

        self.criterion = NtXentLoss(config['batch_size'], config['temperature'])

    class Hash(torch.autograd.Function):
        @staticmethod
        def forward(_, input):
            return input.sign()
        @staticmethod
        def backward(_, grad_output):
            return grad_output

    def forward(self, imgi, imgj, device):
        imgi, _ = self.vit(imgi)
        # imgi = self.vgg.features(imgi)
        # imgi = imgi.view(imgi.size(0), -1)
        # imgi = self.vgg.classifier(imgi)

        prob_i = torch.sigmoid(self.encoder(imgi))
        z_i = CIBHash.Hash.apply(prob_i - 0.5)

        imgj, _ = self.vit(imgj)
        # imgj = self.vgg.features(imgj)
        # imgj = imgj.view(imgj.size(0), -1)
        # imgj = self.vgg.classifier(imgj)

        prob_j = torch.sigmoid(self.encoder(imgj))
        z_j = CIBHash.Hash.apply(prob_j - 0.5)

        kl_loss = (self.compute_kl(prob_i, prob_j) + self.compute_kl(prob_j, prob_i)) / 2
        contra_loss = self.criterion(z_i, z_j, device)
        loss = contra_loss + self.kl_weight * kl_loss

        return {'loss': loss, 'contra_loss': contra_loss, 'kl_loss': kl_loss}

    def encode_discrete(self, x):
        x, _ = self.vit(x)
        # x = self.vgg.features(x)
        # x = x.view(x.size(0), -1)
        # x = self.vgg.classifier(x)

        prob = torch.sigmoid(self.encoder(x))
        z = CIBHash.Hash.apply(prob - 0.5)

        return z

    def compute_kl(self, prob, prob_v):
        prob_v = prob_v.detach()
        # prob = prob.detach()

        kl = prob * (torch.log(prob + 1e-8) - torch.log(prob_v + 1e-8)) + (1 - prob) * (torch.log(1 - prob + 1e-8 ) - torch.log(1 - prob_v + 1e-8))
        kl = torch.mean(torch.sum(kl, axis = 1))
        return kl


def trainer(config, bit):
    Best_mAP = 0

    train_logfile = open(os.path.join(config['logs_path'], 'train_log.txt'), 'a')
    train_logfile.write(f"***** {config['info']} - {config['backbone']} - {bit}bit *****\n\n")

    """DataLoader"""
    if "cifar" in config['dataset']:
        data = CIB_CIFAR_DataLoader(config['dataset'])
        train_loader, test_loader, _, database_loader = data.get_loaders(
            config['batch_size'], 8, 
            shuffle_train=True, get_test=False
        )
    else:
        train_loader, test_loader, database_loader = get_data_CIB(config)

    """Model"""
    device = torch.device('cuda')
    net = CIBHash(bit, config)
    net = net.to(device)

    """Optimizer Setting"""
    optimizer = config["optimizer"]["type"](net.encoder.parameters(), **(config["optimizer"]["optim_params"]))

    """Data Parallel"""
    net = torch.nn.DataParallel(net)

    for epoch in range(config["epoch"]):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("%s-%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], config["backbone"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()
        train_loss = 0
        con_loss = 0
        kl_loss = 0

        for image1, image2, _ in train_loader:
            image1 = image1.to(device)
            image2 = image2.to(device)

            optimizer.zero_grad()

            loss = net(image1, image2, device)

            train_loss += loss['loss'].item()
            con_loss += loss['contra_loss'].item()
            kl_loss += loss['kl_loss'].item()

            loss['loss'].backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)
        con_loss = con_loss / len(train_loader)
        kl_loss = kl_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.5f | con_loss:%.5f | kl_loss:%.5f" % (train_loss, con_loss, kl_loss))

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
    
    config = CIBHash_get_config(start_time)
    save_config(config, config["logs_path"])
    
    for bit in config["bit_list"]:
        trainer(config, bit)
