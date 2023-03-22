import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


class ImageList(object):

    def __init__(self, data_path, image_list, transform):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


def image_transform(resize_size, crop_size, data_set, cfg_info):
    if data_set == "train_dataset":

        if cfg_info in ["IPHash", "CIBHash"]:

            color_jitter = transforms.ColorJitter(0.4,0.4,0.4,0.1)
            return transforms.Compose([transforms.RandomResizedCrop(size = crop_size, scale=(0.5, 1.0)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p = 0.7),
                                            transforms.RandomGrayscale(p  = 0.2),
                                            GaussianBlur(3),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                            ])
        
        else:
            step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
            return transforms.Compose([transforms.Resize(resize_size)]
                                            + step +
                                            [transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ])

    else:
        return transforms.Compose([
                                    transforms.Resize((resize_size, resize_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),                             
                                ])


def My_dataset(config):
    dsets = dict()
    data_config = config["data_list"]

    for data_set in ["train_dataset", "test_dataset", "database_dataset"]:
        dsets[data_set] = ImageList(config["data_path"],
                                    open(data_config[data_set]).readlines(),
                                    transform=image_transform(config["resize_size"], config["crop_size"], data_set, config['info']))

        print(data_set, len(dsets[data_set]))

    return dsets["train_dataset"], dsets["test_dataset"], dsets["database_dataset"], \
        len(dsets["train_dataset"]), len(dsets["test_dataset"]), len(dsets["database_dataset"])