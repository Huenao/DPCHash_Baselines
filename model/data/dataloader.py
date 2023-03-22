import torch

from .cifar_dataset import MyCIFAR10_dataset
from .dataset import My_dataset


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