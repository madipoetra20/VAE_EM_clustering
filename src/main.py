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

from utils import * 
from model.ma import *

parser = argparse.ArgumentParser(description='Run')

parser.add_argument('--file_path', type=str, default = "run_1")
parser.add_argument('--dataset', type=str, default = "FashionMNIST")
parser.add_argument('--epochs', type=int, default = 300)
parser.add_argument('--n_class', type=int, default = 10)
parser.add_argument('--load_state', type=str2bool, default=False)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--latent_dim', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--lambd_rec', type=float, default=1)
parser.add_argument('--loss', type=str, default="MSE")
parser.add_argument('--VAE_epoch', type=int, default=20)
parser.add_argument('--BatchNorm', type=str2bool, default=False)
parser.add_argument('--Dropout', type=str2bool, default=False)
parser.add_argument('--dropout_prob', type=float, default=0.2)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device
args.input_dim =  784

# Get Data
X, y, X_test, y_test = get_data(args.n_class, args.dataset)

print(args, flush = True)

vae = Transductive_GNN(args.dropout_prob, args.BatchNorm, args.Dropout, args.VAE_epoch, args.loss, args.lambd_rec, args.batch_size, args.learning_rate, args.latent_dim, args.n_class, args.epochs, X, y, X_test, y_test, args.file_path, args.device, args.load_state)
u = vae.solve()


