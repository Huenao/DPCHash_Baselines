import os

import torch.optim as optim

from .utils import config_dataset


def get_config(start_time):
    config = {
        "dataset": "mirflickr",
        # "dataset": "cifar10-1",
        # "dataset": "coco",
        # "dataset": "nuswide_21",
        # "dataset": "nuswide_10",

        "bit_list": [16, 32, 64, 128],

        "info": "WCH",
        "backbone": "ViT-B_16", 
        "pretrained_dir": "pretrained_vit/imagenet21k_imagenet2012_ViT-B_16-224.npz",

        "optimizer": {"type": optim.Adam, 
                      "lr": 1e-5},      
        "epoch": 100,
        "test_map": 5,
        "batch_size": 32,
        "num_workers": 4,
        "logs_path": "logs",

        "resize_size": 256,
        "crop_size": 224,

        "alpha": 0.1,
    }
    config = config_dataset(config)
    config["logs_path"] = os.path.join(config["logs_path"], config['info'], start_time)
    
    if not os.path.exists(config["logs_path"]):
        os.makedirs(config["logs_path"])
    
    if 'cifar' in config["dataset"]:
        config["topK"] = 5000
    else: 
        config["topK"] = 5000

    return config