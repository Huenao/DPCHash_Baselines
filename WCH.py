import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from model.VTS_WCH import CONFIGS, VisionTransformer
from configs import WCH_get_config
from model.data import WCH_CIFAR10_dataloader, WCH_dataloader
from utils.utils import evalModel, save_config

torch.multiprocessing.set_sharing_strategy('file_system')


class CL(nn.Module):
    def __init__(self, config, bit):
        super(CL, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.bit = bit

    def forward(self, h1, h2, weighted):
        logits = torch.einsum('ik,jk->ij', h1, h2)
        logits = logits / self.bit / 0.3

        balance_logits = h1.sum(0) / h1.size(0)
        reg = self.mse(balance_logits, torch.zeros_like(balance_logits)) - self.mse(h1, torch.zeros_like(h1))

        loss = self.ce(logits, weighted) + reg

        return loss
    

def trainer(config, bit):
    Best_mAP = 0

    train_logfile = open(os.path.join(config['logs_path'], 'train_log.txt'), 'a')
    train_logfile.write(f"***** {config['info']} - {config['backbone']} - {bit}bit *****\n\n")

    """DataLoader"""
    if "cifar" in config['dataset']:
        train_loader, test_loader, database_loader, num_train, num_test, num_database = WCH_CIFAR10_dataloader(config)
    else:
        train_loader, test_loader, database_loader, num_train, num_test, num_database = WCH_dataloader(config)

    """Model"""
    device = torch.device('cuda')
    vit_config = CONFIGS[config['backbone']]
    vit_config.pretrained_dir = config['pretrained_dir']

    net = VisionTransformer(vit_config, 224, num_classes=config['n_class'], zero_head=True, hash_bit=bit).to(device)
    net.load_from(np.load(vit_config.pretrained_dir))
    
    criterion = CL(config, bit)

    """Optimizer Setting"""
    optimizer = config["optimizer"]["type"]([{"params": net.parameters(), "lr": config["optimizer"]["lr"]}
                                            ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epoch"])

    """Training"""
    for epoch in range(config["epoch"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("%s-%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], config["backbone"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()
        train_loss = 0

        for image1, image2 in train_loader:
            image1, image2 = image1.to(device), image2.to(device)
            
            optimizer.zero_grad()
            
            h1, h2, weight = net.train_forward(image1, image2)
            loss = criterion(h1, h2, weight)
            
            train_loss += loss.item()
            loss.backward()
            
            optimizer.step()
            
        train_loss = train_loss / len(train_loader)
        scheduler.step()

        print("\b\b\b\b\b\b\b loss:%.5f" % train_loss)

        if (epoch + 1) % config["test_map"] == 0:
            net.eval()
            with torch.no_grad():
                Best_mAP = evalModel(test_loader, database_loader, net, Best_mAP, bit, config, epoch, train_logfile, num_database)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    start_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    setup_seed(2022)
    
    config = WCH_get_config(start_time)
    save_config(config, config["logs_path"])
    
    for bit in config["bit_list"]:
        config["pr_curve_path"] = os.path.join(config["logs_path"], f"pr_curve_{bit}.json")
        trainer(config, bit)

