import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .dataset import GaussianBlur


class ImageList_CIB(object):

    def __init__(self, data_path, image_list, transform, train=False):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform
        self.train = train

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.train:
            imgi = self.transform(img)
            imgj = self.transform(img)
            return imgi, imgj, target
        else:
            img = self.transform(img)
            return img, target

    def __len__(self):
        return len(self.imgs)


def get_transform_CIB(resize_size, crop_size):
    color_jitter = transforms.ColorJitter(0.4,0.4,0.4,0.1)
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(size = crop_size,scale=(0.5, 1.0)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomApply([color_jitter], p = 0.7),
                                        transforms.RandomGrayscale(p  = 0.2),
                                        GaussianBlur(3),
                                        transforms.ToTensor(),
                                        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
                                    ])
    test_transforms = transforms.Compose([
                                    transforms.Resize((resize_size, resize_size)),
                                    transforms.ToTensor(),
                                    # transforms.Normaliz([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])     
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])                                
                                ])
    return train_transforms, test_transforms


def get_data_CIB(config):
    data_config = config["data_list"]
    train_transform, test_transform = get_transform_CIB(config["resize_size"], config["crop_size"])
    train_dataset = ImageList_CIB(config["data_path"],
                                    open(data_config["train_dataset"]).readlines(),
                                    transform=train_transform, train=True)
    

    test_dataset = ImageList_CIB(config["data_path"],
                                    open(data_config["test_dataset"]).readlines(),
                                    transform=test_transform, train=False)

    database_dataset = ImageList_CIB(config["data_path"],
                                    open(data_config["database_dataset"]).readlines(),
                                    transform=test_transform, train=False)
    
    print('train_dataset', len(train_dataset))
    print('test_dataset', len(test_dataset))
    print('database_dataset', len(database_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config['batch_size'],
                                               shuffle=True, 
                                               num_workers=config['num_workers'])
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=config['batch_size'],
                                              shuffle=False, 
                                              num_workers=config['num_workers'])
    database_loader = torch.utils.data.DataLoader(database_dataset,
                                                  batch_size=config['batch_size'],
                                                  shuffle=False, 
                                                  num_workers=config['num_workers'])

    return train_loader, test_loader, database_loader, \
           len(train_dataset), len(test_dataset), len(database_dataset)