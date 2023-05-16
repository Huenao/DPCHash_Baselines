import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import BiHalf_get_config
from model.data import my_dataloader
from model.vit import CONFIGS, VisionTransformer
from utils.utils import evalModel, save_config

torch.multiprocessing.set_sharing_strategy('file_system')


# Deep Unsupervised Image Hashing by Maximizing Bit Entropy(AAAI2021)
# paper [Deep Unsupervised Image Hashing by Maximizing Bit Entropy](https://arxiv.org/pdf/2012.12334.pdf)
# code [Deep-Unsupervised-Image-Hashing](https://github.com/liyunqianggyn/Deep-Unsupervised-Image-Hashing)

class BiHalfModelUnsupervised(nn.Module):
    def __init__(self, bit, config):
        super(BiHalfModelUnsupervised, self).__init__()

        vit_config = CONFIGS[config['backbone']]
        vit_config.pretrained_dir = config['pretrained_dir']

        self.vit = VisionTransformer(vit_config, 224, num_classes=1000, zero_head=False, vis=True)
        self.vit.load_from(np.load(vit_config.pretrained_dir))
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.eval()

        self.fc_encode = nn.Linear(vit_config.hidden_size, bit)

        # self.vgg = models.vgg16(pretrained=True)
        # self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:6])
        # for param in self.vgg.parameters():
        #     param.requires_grad = False

        # self.fc_encode = nn.Linear(4096, bit)

    class Hash(torch.autograd.Function):
        @staticmethod
        def forward(ctx, U):
            # Yunqiang for half and half (optimal transport)
            _, index = U.sort(0, descending=True)
            N, D = U.shape
            B_creat = torch.cat((torch.ones([int(N / 2), D]), -torch.ones([N - int(N / 2), D]))).cuda()
            B = torch.zeros(U.shape).cuda().scatter_(0, index, B_creat)
            ctx.save_for_backward(U, B)
            return B

        @staticmethod
        def backward(ctx, g):
            U, B = ctx.saved_tensors
            add_g = (U - B) / (B.numel())
            grad = g + config["gamma"] * add_g
            return grad

    def forward(self, x):
        x, _ = self.vit(x)
        # x = self.vgg.features(x)
        # x = x.view(x.size(0), -1)
        # x = self.vgg.classifier(x)

        h = self.fc_encode(x)
        if not self.training:
            return h.sign()
        else:
            b = BiHalfModelUnsupervised.Hash.apply(h)
            target_b = F.cosine_similarity(b[:x.size(0) // 2], b[x.size(0) // 2:])
            target_x = F.cosine_similarity(x[:x.size(0) // 2], x[x.size(0) // 2:])
            loss = F.mse_loss(target_b, target_x)
            return loss


def trainer(config, bit):
    Best_mAP = 0

    train_logfile = open(os.path.join(config['logs_path'], 'train_log.txt'), 'a')
    train_logfile.write(f"***** {config['info']} - {config['backbone']} - {bit}bit *****\n\n")

    """DataLoader"""
    train_loader, test_loader, database_loader, num_train, num_test, num_database = my_dataloader(config)
   
    """Model"""
    device = torch.device('cuda')
    net = BiHalfModelUnsupervised(bit, config)
    net = net.to(device)

    """Optimizer Setting"""
    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    """Data Parallel"""
    net = torch.nn.DataParallel(net)

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
            image = image.to(device)
            optimizer.zero_grad()

            loss = net(image)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)
        print("\b\b\b\b\b\b\b loss:%.5f" % (train_loss))
        train_logfile.write('Train | %s-%s[%2d/%2d][%s] bit:%d, dataset:%s | Loss: %.5f \n'% 
                    (config["info"], config["backbone"], epoch+1, config["epoch"], current_time, bit, config["dataset"], train_loss))

        if (epoch + 1) % config["test_map"] == 0:
            net.eval()
            with torch.no_grad():
                Best_mAP = evalModel(test_loader, database_loader, net, Best_mAP, bit, config, epoch+1, train_logfile)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    start_time = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))
    setup_seed(2022)
    
    config = BiHalf_get_config(start_time)
    save_config(config, config["logs_path"])
    
    for bit in config["bit_list"]:
        trainer(config, bit)
