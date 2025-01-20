import sys
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from torch.distributions import Normal
from torch.optim import Adam

from scipy.special import logsumexp
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.ndimage import rotate
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.decomposition import PCA


def get_data(n_classes, dataset="MNIST"):
    """
    Load dataset and return data and labels.

    Args:
        n_classes (int): Number of classes to include.
        dataset (str): Name of the dataset to load. Options are "MNIST", "FashionMNIST", "STL-10".

    Returns:
        x_train (np.ndarray): Data samples.
        y_train (np.ndarray): Corresponding labels.
    """
    if dataset == "MNIST":
        print("MNIST", flush = True)
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True)
    elif dataset == "FashionMNIST":
        print("Fashion", flush = True)
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True)
    else:
        raise ValueError(f"Dataset {dataset} not recognized.")

    # Normalize train and test data to [0, 1]
    x_train = np.array(trainset.data, dtype=np.float32) / 255.0
    y_train = np.array(trainset.targets)
    x_test = np.array(testset.data, dtype=np.float32) / 255.0
    y_test = np.array(testset.targets)


    # Select digits
    numbers = list(range(10))
    if n_classes == 10:
        digits = numbers  # Use all digits
    else:
        digits = random.sample(numbers, n_classes)
    print(f"Selected digits: {digits}", flush=True)

    # Filter data for selected digits
    mask = np.isin(y_train, digits)
    x_train = x_train[mask]
    y_train = y_train[mask]

    mask_test = np.isin(y_test, digits)
    x_test = x_test[mask_test]
    y_test = y_test[mask_test]

    return x_train, y_train, x_test, y_test

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')