import numpy as np
import torch
from PIL import Image

from .dataset import ImageList
from .WCH_cifar_data import test_transform, train_transform


class ImageList_WCH(object):

    def __init__(self, data_path, image_list, train_transform):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.train_transform = train_transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img1 = self.train_transform(img)
        img2 = self.train_transform(img)
        return img1, img2

    def __len__(self):
        return len(self.imgs)
    

def WCH_dataloader(config):
    dsets = dict()
    data_config = config["data_list"]

    for data_set in ["train_dataset", "test_dataset", "database_dataset"]:
        if data_set == "train_dataset":
            dsets[data_set] = ImageList_WCH(config["data_path"],
                                            open(data_config[data_set]).readlines(),
                                            train_transform=train_transform)
        else:
            dsets[data_set] = ImageList(config["data_path"],
                                        open(data_config[data_set]).readlines(),
                                        transform=test_transform)

        print(data_set, len(dsets[data_set]))

    
    train_loader = torch.utils.data.DataLoader(dataset = dsets["train_dataset"],
                                               batch_size = config['batch_size'],
                                               shuffle = True,
                                               num_workers = config['num_workers'])

    test_loader = torch.utils.data.DataLoader(dataset = dsets["test_dataset"],
                                              batch_size = config['batch_size'],
                                              shuffle = False,
                                              num_workers = config['num_workers'])

    database_loader = torch.utils.data.DataLoader(dataset = dsets["database_dataset"],
                                                  batch_size = config['batch_size'],
                                                  shuffle = False,
                                                  num_workers = config['num_workers'])
    
    return train_loader, test_loader, database_loader, \
        len(dsets["train_dataset"]), len(dsets["test_dataset"]), len(dsets["database_dataset"])