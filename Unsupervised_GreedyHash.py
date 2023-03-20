import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import GreedyHash_get_config
from model.datasets import My_dataset, MyCIFAR10_dataset
from model.vit import CONFIGS, VisionTransformer
from utils.utils import evalModel, save_config

torch.multiprocessing.set_sharing_strategy('file_system')

# GreedyHash(NIPS2018)
# paper [Greedy Hash: Towards Fast Optimization for Accurate Hash Coding in CNN](https://papers.nips.cc/paper/7360-greedy-hash-towards-fast-optimization-for-accurate-hash-coding-in-cnn.pdf)
# code [GreedyHash](https://github.com/ssppp/GreedyHash)


class GreedyHashModelUnsupervised(nn.Module):
    def __init__(self, bit, config):
        super(GreedyHashModelUnsupervised, self).__init__()

        vit_config = CONFIGS[config['backbone']]
        vit_config.pretrained_dir = config['pretrained_dir']

        self.vit = VisionTransformer(vit_config, 224, num_classes=1000, zero_head=False, vis=True)
        self.vit.load_from(np.load(vit_config.pretrained_dir))
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.eval()

        self.fc_encode = nn.Linear(vit_config.hidden_size, bit)

        # self.vgg = models.vgg16(pretrained=True)s
        # self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:6])
        # for param in self.vgg.parameters():
        #     param.requires_grad = False

        # self.fc_encode = nn.Linear(4096, bit)

    class Hash(torch.autograd.Function):
        @staticmethod
        def forward(_, input):
            return input.sign()
        @staticmethod
        def backward(_, grad_output):
            return grad_output

    def forward(self, x):
        x, _ = self.vit(x)
        # x = self.vgg.features(x)
        # x = x.view(x.size(0), -1)
        # x = self.vgg.classifier(x)

        h = self.fc_encode(x)
        b = GreedyHashModelUnsupervised.Hash.apply(h)
        if not self.training:
            return b
        else:
            target_b = F.cosine_similarity(b[:x.size(0) // 2], b[x.size(0) // 2:])
            target_x = F.cosine_similarity(x[:x.size(0) // 2], x[x.size(0) // 2:])
            loss1 = F.mse_loss(target_b, target_x)
            loss2 = config["alpha"] * (h.abs() - 1).pow(3).abs().mean()
            return loss1 + loss2
        

def my_dataloader(config):
    if "cifar" in config["dataset"]:
        train_dataset, test_dataset, database_dataset, num_train, num_test, num_database = MyCIFAR10_dataset(config)
    else:
        train_dataset, test_dataset, database_dataset, num_train, num_test, num_database = My_dataset(config)

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                               batch_size = config['batch_size'],
                                               shuffle = True,
                                               num_workers = config['num_workers'])

    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                              batch_size = config['batch_size'],
                                              shuffle = False,
                                              num_workers = config['num_workers'])

    database_loader = torch.utils.data.DataLoader(dataset = database_dataset,
                                                  batch_size = config['batch_size'],
                                                  shuffle = False,
                                                  num_workers = config['num_workers'])
    return train_loader, test_loader, database_loader, num_train, num_test, num_database


def trainer(config, bit):
    Best_mAP = 0

    train_logfile = open(os.path.join(config['logs_path'], 'train_log.txt'), 'a')
    train_logfile.write(f"***** {config['info']} - {config['backbone']} - {bit}bit *****\n\n")

    """DataLoader"""
    train_loader, test_loader, database_loader, num_train, num_test, num_database = my_dataloader(config)
   
    """Model"""
    net = GreedyHashModelUnsupervised(bit, config)
    # net = torch.nn.DataParallel(net)
    net = net.cuda()

    """Optimizer Setting"""
    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    """Training"""
    for epoch in range(config["epoch"]):

        """lr decrease"""
        lr = config["optimizer"]["optim_params"]["lr"] * (0.1 ** (epoch // config["optimizer"]["epoch_lr_decrease"]))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("%s-%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], config["backbone"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()
        train_loss = 0
        for image, _, _ in train_loader:
            image = image.cuda()
            optimizer.zero_grad()

            loss = net(image)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)
        print("\b\b\b\b\b\b\b loss:%.5f" % (train_loss))

        if (epoch + 1) % config["test_map"] == 0:
            net.eval()
            with torch.no_grad():
                Best_mAP = Best_mAP = evalModel(test_loader, database_loader, net, Best_mAP, bit, config, epoch, train_logfile)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    start_time = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))
    setup_seed(2022)
    
    config = GreedyHash_get_config(start_time)
    save_config(config, config["logs_path"])
    os.environ['CUDA_VISIBLE_DEVICES'] = config['device']
    
    for bit in config["bit_list"]:
        trainer(config, bit)
