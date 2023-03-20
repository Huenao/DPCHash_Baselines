import os

import torch.optim as optim

from .utils import config_dataset


def get_config(start_time):
    config = {
        "dataset": "cifar10",
        # "dataset": "cifar10-1",
        # "dataset": "cifar10-2",
        # "dataset": "coco",
        # "dataset": "nuswide_21",

        "bit_list": [16, 32, 64],

        "info": "IPHash",
        "backbone": "ViT-B_16", 
        "pretrained_dir": "pretrained_vit/imagenet21k_imagenet2012_ViT-B_16.npz",

        "optimizer": {"type": optim.Adam, 
                      "lr": 0.001, 
                      "backbone_lr": 1e-5},
        "device": '0, 1',        
        "epoch": 100,
        "test_map": 5,
        "batch_size": 64,
        "num_workers": 4,
        "logs_path": "logs",

        "resize_size": 256,
        "crop_size": 224,

        "temperature": [20],
        "mask_ratio": 0.25,
        "alpha": 0.1,
        "beta": 1.,
        "gamma": 0.1,
    }
    config = config_dataset(config)
    config["logs_path"] = os.path.join(config["logs_path"], start_time)
    if not os.path.exists(config["logs_path"]):
        os.makedirs(config["logs_path"])
    
    if 'cifar' in config["dataset"]:
        config["topK"] = 1000
    else: 
        config["topK"] = 5000

    return config
