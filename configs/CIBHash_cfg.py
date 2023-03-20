import os

import torch.optim as optim

from .utils import config_dataset


def get_config(start_time):
    config = {
        "dataset": "cifar10",
        # "dataset": "coco",
        # "dataset": "nuswide_21",
        
        "weight": 0.001,
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 0.001}},

        "info": "[CIBHash]",
        "resize_size": 224,
        "crop_size": 224,
        "batch_size": 64,

        "epoch": 100,
        "test_map": 5,

        "bit_list": [32],
        "temperature": 0.3,
        "save_path": "save/CIBHash_NUS"
    }
    config = config_dataset(config)

    config["logs_path"] = os.path.join(config["logs_path"], config['info'], start_time)
    if not os.path.exists(config["logs_path"]):
        os.makedirs(config["logs_path"])

    if 'cifar' in config["dataset"]:
        config["topK"] = 1000
    else: 
        config["topK"] = 5000

    return config