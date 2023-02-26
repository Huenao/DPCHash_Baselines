import torch
import torch.optim as optim

from utils.tools import *
from IPHash import IPHash

def get_config():
    config = {
        "alpha": 0.1,
        "optimizer": {"type": optim.Adam, "lr": 0.001, "backbone_lr": 1e-5},

        "info": "[IPHash]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        "net": IPHash,
        # "dataset": "cifar10-1",  
        "dataset": "coco",
        # "dataset": "nuswide_21",
        "epoch": 100,
        "test_map": 10,
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:4"),
        "bit_list": [64],
        "save_path": "save/IPHash",
        "temperature": [20],
        "mask_ratio": 0.25,
        "beta": 1.,
        "gamma": 0.1,
    }
    config = config_dataset(config)
    if config["dataset"] == "cifar10-1":
        config["topK"] = 1000
    else: 
        config["topK"] = 5000

    return config
