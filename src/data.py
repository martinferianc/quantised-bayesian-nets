import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import random
import torchvision
import logging
import os
from os import path
from sklearn.model_selection import KFold
import pandas as pd
import zipfile
import urllib.request
from src.utils import BRIGHTNESS_LEVELS, SHIFT_LEVELS, ROTATION_LEVELS

CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
MNIST_MEAN, MNIST_STD = (0,), (1,)

class UCIDatasets():
    def __init__(self,  name,  data_path="", n_splits = 10):
        self.datasets = {
            "housing": "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
            "concrete": "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
            "energy": "http://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
            "power": "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip",
            "wine": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            "yacht": "http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data"}
        self.data_path = data_path
        self.name = name
        self.n_splits = n_splits
        self._load_dataset()

    
    def _load_dataset(self):
        if self.name not in self.datasets:
            raise Exception("Not known dataset!")
        if not path.exists(self.data_path+"UCI"):
            os.mkdir(self.data_path+"UCI")

        url = self.datasets[self.name]
        file_name = url.split('/')[-1]
        if not path.exists(self.data_path+"UCI/" + file_name):
            urllib.request.urlretrieve(
                self.datasets[self.name], self.data_path+"UCI/" + file_name)
        data = None

        if self.name == "housing":
            data = pd.read_csv(self.data_path+'UCI/housing.data',
                        header=0, delimiter="\s+").values
            self.data = data
        elif self.name == "concrete":
            data = pd.read_excel(self.data_path+'UCI/Concrete_Data.xls',
                               header=0).values
            self.data = data
        elif self.name == "energy":
            data = pd.read_excel(self.data_path+'UCI/ENB2012_data.xlsx',
                                 header=0).values
            self.data = data
        elif self.name == "power":
            zipfile.ZipFile(self.data_path +"UCI/CCPP.zip").extractall(self.data_path +"UCI/")
            data = pd.read_excel(self.data_path+'UCI/CCPP/Folds5x2_pp.xlsx', header=0).values
            self.data = data
        elif self.name == "wine":
            data = pd.read_csv(self.data_path + 'UCI/winequality-red.csv',
                               header=1, delimiter=';').values
            self.data = data

        elif self.name == "yacht":
            data = pd.read_csv(self.data_path + 'UCI/yacht_hydrodynamics.data',
                               header=1, delimiter='\s+').values
            self.data = data
            
        kf = KFold(n_splits=self.n_splits)
        self.in_dim = data.shape[1] - 1
        self.out_dim = 1
        self.data_splits = kf.split(data)
        self.data_splits = [(idx[0], idx[1]) for idx in self.data_splits]

    def get_split(self, split=-1, train=True):
        if split == -1:
            split = 0
        if 0<=split and split<self.n_splits: 
            train_index, test_index = self.data_splits[split]
            x_train, y_train = self.data[train_index,
                                    :self.in_dim], self.data[train_index, self.in_dim:]
            x_test, y_test = self.data[test_index, :self.in_dim], self.data[test_index, self.in_dim:]
            x_means, x_stds = x_train.mean(axis=0), x_train.var(axis=0)**0.5
            y_means, y_stds = y_train.mean(axis=0), y_train.var(axis=0)**0.5
            x_train = (x_train - x_means)/x_stds
            y_train = (y_train - y_means)/y_stds
            x_test = (x_test - x_means)/x_stds
            y_test = (y_test - y_means)/y_stds
            if train:
                inps = torch.from_numpy(x_train).float()
                tgts = torch.from_numpy(y_train).float()
                train_data = torch.utils.data.TensorDataset(inps, tgts)
                return train_data
            else:
                inps = torch.from_numpy(x_test).float()
                tgts = torch.from_numpy(y_test).float()
                test_data = torch.utils.data.TensorDataset(inps, tgts)
                return test_data

class HorizontalTranslate(object):
    def __init__(self, distance, img_size):
        self.distance = distance
        self.img_size = img_size

    def __call__(self, image):
        img_size = self.img_size
        dx = float(self.distance * img_size[0])
        tx = int(round(dx))
        ty = 0
        translations = (tx, ty)
        return torchvision.transforms.functional.affine(image, 0, translations, 1.0, 0, resample=0, fillcolor=0)


def regression_function(X, noise=True):
    w = 2.
    sigma = 1.
    b = 8.
    Y = X.dot(w)+b
    if noise:
        return Y + np.reshape(sigma*np.random.normal(0., 1., len(X)), (len(X), 1))
    else:
        return Y


def regression_data_generator(N_points=100, X = None, noise=True):
    if X is None:
        X = np.reshape(np.random.randn(N_points, 1), (N_points, 1))
    
    Y = regression_function(X, noise)
    return X, Y


def get_train_loaders(args, split=-1):
    assert args.valid_portion >= 0 and args.valid_portion < 1.
    train_data = None
    if args.dataset == "mnist":
        train_transform = transforms.Compose([transforms.ToTensor(), \
                                transforms.Normalize(MNIST_MEAN, MNIST_STD)])

        train_data = datasets.MNIST(root=args.data, train=True, 
                                    download=True, transform=train_transform)
    elif args.dataset == "cifar":
        train_transform =[]
        train_transform.append(transforms.RandomCrop(32, padding=4))
        train_transform.append(transforms.RandomHorizontalFlip())
        train_transform.append(transforms.ToTensor())
        train_transform.append(transforms.Normalize(CIFAR_MEAN, CIFAR_STD))

        train_transform = transforms.Compose(train_transform)

        train_data = datasets.CIFAR10(root=args.data, train=True, 
                                    download=True, transform=train_transform)
    elif "regression" in args.dataset:
        if args.dataset == "regression_synthetic":
            X, Y = regression_data_generator(N_points=1000)
            inps = torch.from_numpy(X).float()
            tgts = torch.from_numpy(Y).float()
            train_data = torch.utils.data.TensorDataset(inps, tgts)
        else:
            train_data = UCIDatasets(args.dataset.split("_")[-1], args.data).get_split(split, train=True)
    else:
        raise NotImplementedError("Other datasets not implemented")
    return get_train_split_loaders(args.valid_portion, train_data, args.batch_size, args.num_workers)



def get_train_split_loaders(valid_portion, train_data, batch_size, num_workers=0):
    num_train = len(train_data)
    indices = list(range(len(train_data)))
    indices = random.sample(indices, num_train)
    valid_split = int(
        np.floor((valid_portion) * num_train))   # 40k
    valid_idx, train_idx = indices[:valid_split], indices[valid_split:]

    train_sampler = SubsetRandomSampler(train_idx)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler,
        pin_memory=True, num_workers=num_workers)
    
        
    valid_loader = None
    if valid_portion>0.0:
        valid_sampler = SubsetRandomSampler(valid_idx)

        valid_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, sampler=valid_sampler, 
            pin_memory=True, num_workers=num_workers)
    logging.info('### Train size: {}, Validation size: {} ###'.format(len(train_idx), len(valid_idx)))

    return train_loader, valid_loader


def get_test_loader(args, distortion=None, level=-1, split = -1):
    test_data = None
    if args.dataset == "mnist":
        test_transform = []
        if distortion=="rotation":
            test_transform.append(transforms.RandomAffine(ROTATION_LEVELS[level], translate=None, scale=None, shear=None, resample=False, fillcolor=0))
        elif distortion=="shift":
            test_transform.append(HorizontalTranslate(SHIFT_LEVELS[level], [28, 28]))
        elif distortion=="brightness":
            test_transform.append(torchvision.transforms.ColorJitter(brightness=BRIGHTNESS_LEVELS[level]))

        test_transform += [transforms.ToTensor(),
                          transforms.Normalize((0.0,), (1.,))]

        test_transform = transforms.Compose(test_transform)

        test_data = datasets.MNIST(root=args.data, train=False,
                                   download=True, transform=test_transform)
    elif args.dataset == "cifar":
        test_transform = []

        if distortion=="rotation":
            test_transform.append(transforms.RandomAffine(ROTATION_LEVELS[level], translate=None, scale=None, shear=None, resample=False, fillcolor=0))
        elif distortion=="shift":
            test_transform.append(HorizontalTranslate(
                SHIFT_LEVELS[level], [32, 32]))
        elif distortion == "brightness":
            test_transform.append(torchvision.transforms.ColorJitter(
                brightness=BRIGHTNESS_LEVELS[level]))

        test_transform += [transforms.ToTensor(),
                          transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]

        test_transform = transforms.Compose(test_transform)

        test_data = datasets.CIFAR10(root=args.data, train=False,
                                   download=True, transform=test_transform)

    elif "regression" in args.dataset:
        if args.dataset == "regression_synthetic":
            X, Y = regression_data_generator(N_points = 1000, noise=False)
            inps = torch.from_numpy(X).float()
            tgts = torch.from_numpy(Y).float()
            test_data = torch.utils.data.TensorDataset(inps, tgts)
        else:
            test_data = UCIDatasets(args.dataset.split("_")[-1], args.data).get_split(split, train=False)

    elif args.dataset == "random_mnist":
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.0,), (1.,))
        ])
        test_data = datasets.FashionMNIST(root=args.data, train=False,
                                    download=True, transform=test_transform)
    elif args.dataset == "random_cifar":
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        ])        
        test_data = datasets.SVHN(root=args.data, split='test',
                                    download=True, transform=test_transform)
    else:
        raise NotImplementedError("Other datasets not implemented")
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                              shuffle=False, pin_memory=True, num_workers=args.num_workers)
    logging.info('### Test size: {} ###'.format(len(test_loader.dataset)))
    return test_loader
