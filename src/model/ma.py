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
from torch.optim.lr_scheduler import StepLR

from scipy.special import logsumexp
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.ndimage import rotate
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score

# Structure of VAE is from https://github.com/GuHongyang/VaDE-pytorch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return idx, self.X[idx], self.y[idx]

def calc_accuracy(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1).cpu().numpy()
    y_true = y_true.astype(np.int64)    

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size

def block(input_dim, output_dim, BatchNorm = False, Dropout = False, dropout_prob=0.2):
    if BatchNorm == False and Dropout == False:
        layers=[
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(0.2, inplace=True),
        ]
    elif BatchNorm == False and Dropout == True:
        layers=[
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_prob)
        ]
    elif BatchNorm == True and Dropout == False:
        layers=[
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        ]
    else:
        layers=[
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_prob)
        ]
    return layers

class Encoder(nn.Module):
    def __init__(self, dropout_prob, latent_dim, BatchNorm, Dropout, input_dim=784, hid_dim=[500]):
        super(Encoder, self).__init__()

        # Dynamically create encoder layers based on hid_dim
        layers = []
        current_dim = input_dim
        for hid_dim_layer in hid_dim:
            layers.extend(block(current_dim, hid_dim_layer, BatchNorm, Dropout, dropout_prob))
            current_dim = hid_dim_layer
        
        self.encoder = nn.Sequential(*layers)
        self.FC_mean = nn.Linear(hid_dim[-1], latent_dim)
        self.FC_var = nn.Linear(hid_dim[-1], latent_dim)

    def forward(self, x):
        x = x.view(len(x), -1)
        e = self.encoder(x)
        mu = self.FC_mean(e)
        log_var = self.FC_var(e)

        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, dropout_prob, latent_dim, BatchNorm, Dropout, input_dim=784, hid_dim=[500]):
        super(Decoder, self).__init__()

        # Dynamically create decoder layers based on hid_dim
        layers = []
        current_dim = latent_dim
        for hid_dim_layer in hid_dim:
            layers.extend(block(current_dim, hid_dim_layer, BatchNorm, Dropout, dropout_prob))
            current_dim = hid_dim_layer

        # Add final output layer and sigmoid
        layers.append(nn.Linear(hid_dim[-1], input_dim))
        if input_dim == 784:
            layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        x_hat = self.decoder(z)
        return x_hat

class Model(nn.Module):
    def __init__(self, Encoder, Decoder, device):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.device = device
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)        
        z = mean + var * epsilon                         
        return z
        
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) 
        x_hat = self.Decoder(z)
        
        return x_hat, mean, log_var

class Transductive_GNN:
    def __init__(self, dropout_prob, BatchNorm, Dropout, VAE_epoch, loss, lambd_rec, batch_size, lr, latent_dim, n_class, n_iter, X, y, X_test, y_test, file_path, device = "cpu", load_state = False):
        self.device = device
        self.file_path = file_path
        self.folder = "answer"
        self.batch_size = batch_size
        self.load_state = load_state
        self.lambd_rec = lambd_rec
        self.lr = lr
        self.epochs = n_iter
        self.VAE_epochs = VAE_epoch
        self.monte_carlo_sample = 10
        self.loss = loss
        self.BatchNorm = BatchNorm
        self.Dropout = Dropout
        self.dropout_prob = dropout_prob
        
        # number of class/cluster/digits
        self.n_class = n_class
        # number of iterations
        self.n_iter = n_iter
        # training data
        self.X = torch.tensor(X).float().to(self.device)
        self.y = y
        self.X_test = torch.tensor(X_test).float().to(self.device)
        self.y_test = y_test
        self.y_test_pred = None
        # number of sample in the training data
        self.n_sample = X.shape[0]
        # the dimension of each sample in the training data
        flatten_X = self.X.view(len(self.X), -1)
        self.input_dim = flatten_X.shape[1]
        # latent dimension
        self.latent_dim = latent_dim
        
        # Initialize the weights/soft assignment
        self.weights = torch.rand(self.n_class, self.n_sample) 
        self.weights = self.weights/self.weights.sum(axis = 0)


        self.encoders = []
        self.decoders = []
        self.optimizers = []

        all_param = []
        for i in range(n_class):
            encoder = Encoder(self.dropout_prob, self.latent_dim, self.BatchNorm, self.Dropout, self.input_dim).to(self.device)
            decoder = Decoder(self.dropout_prob, self.latent_dim, self.BatchNorm, self.Dropout, self.input_dim).to(self.device)
            
            all_param.extend(list(encoder.parameters()))
            all_param.extend(list(decoder.parameters()))

            self.encoders.append(encoder)
            self.decoders.append(decoder)
        
        self.optimizer = Adam(all_param, lr=self.lr)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.9)


    def loss_function(self, x, x_hat, mean, log_var, weights):
        if self.loss == "BCE":
            reproduction_loss = self.lambd_rec * nn.functional.binary_cross_entropy(x_hat, x, reduction='none')
        else:
            reproduction_loss = self.lambd_rec * 0.5 * nn.functional.mse_loss(x_hat, x, reduction='none')
        reproduction_loss = torch.sum(reproduction_loss * weights.unsqueeze(1).to(self.device))
        KLD = - 0.5 * torch.sum((1 + log_var - mean.pow(2) - log_var.exp()) * weights.unsqueeze(1).to(self.device))
        loss = (reproduction_loss + KLD)
        return loss

    def train_model(self, data_loader, weights_in):
        for encoder, decoder in zip(self.encoders, self.decoders):
            encoder.train()
            decoder.train()

        for epoch in range(self.VAE_epochs):
            overall_loss = 0
            for batch_idx, (indices, x, y) in enumerate(data_loader):
                weights_batch = weights_in[:,indices].to(self.device)

                self.optimizer.zero_grad()
                batch_loss = 0
                for k in range(self.n_class):
                    encoder = self.encoders[k]
                    decoder = self.decoders[k]
                    model = Model(encoder, decoder, self.device).to(self.device)
                    
                    # Forward pass
                    x_hat, mean, log_var = model(x)
                    x = x.view(x.size(0), -1).to(self.device) 
                    x_hat = x_hat.view(x.size(0), -1).to(self.device) 
                    
                    # Compute loss for this class
                    loss_k = self.loss_function(x, x_hat, mean, log_var, weights_batch[k])
                    batch_loss += loss_k

                batch_loss.backward()
                self.optimizer.step()

    def maximization_step(self, iteration):
        train_loader = DataLoader(dataset=CustomDataset(self.X, torch.tensor(self.y)), batch_size=self.batch_size, shuffle=True)
        self.train_model(train_loader, self.weights)

        for k in range(self.n_class):
            decoder = self.decoders[k]
            
            # Save the trained models, encoders, and decoders
            if iteration % 10 == 0:
                # torch.save(encoder.state_dict(), f'{self.folder}/{self.file_path}/encoder_{k}.pth')
                torch.save(decoder.state_dict(), f'{self.folder}/{self.file_path}/decoder_{k}.pth')
    
    
    
    def estimation_step(self, iteration):
        chunk_size = 20000        
        data_loader = DataLoader(dataset=CustomDataset(self.X, torch.tensor(self.y)), batch_size=chunk_size, shuffle=False)

        logits = []
        for i in range(self.n_class):
            model_logits = []
            for batch_idx, (indices, x_chunk, y) in enumerate(data_loader):
                x_chunk = x_chunk.to(self.device)
                
                with torch.no_grad():
                    log_px = self.ELBO(x_chunk, self.encoders[i], self.decoders[i])
                model_logits.append(log_px)
            
            model_logits_tensor = torch.cat(model_logits, dim=0)
            logits.append(model_logits_tensor)

        logits_tensor = torch.stack(logits, dim=0).to(self.device) 
        self.weights = logits_tensor.softmax(0)


    def test(self, iteration):
        chunk_size = 20000        
        data_loader = DataLoader(dataset=CustomDataset(self.X_test, torch.tensor(self.y_test)), batch_size=chunk_size, shuffle=False)

        logits = []
        for i in range(self.n_class):
            model_logits = []
            for batch_idx, (indices, x_chunk, y) in enumerate(data_loader):
                x_chunk = x_chunk.to(self.device)
                
                with torch.no_grad():
                    log_px = self.ELBO(x_chunk, self.encoders[i], self.decoders[i])
                model_logits.append(log_px)
            
            model_logits_tensor = torch.cat(model_logits, dim=0)
            logits.append(model_logits_tensor)

        self.y_test_pred = torch.stack(logits, dim=0).to(self.device)         
        

    def ELBO(self, x, encoder, decoder):
        batch_size = x.size(0)
        num_samples = self.monte_carlo_sample
        x = x.view(batch_size, -1).to(self.device)
        
        # Encode x
        z_mean, z_log_var = encoder(x)
        
        # Sample z from the approximate posterior q(z|x)
        std = torch.exp(0.5 * z_log_var)
        qz = torch.distributions.Normal(z_mean, std)
        z_samples = qz.rsample([num_samples])  # Shape: [num_samples, batch_size, latent_dim]
        
        # Decode z_samples
        z_samples = z_samples.view(-1, self.latent_dim)  # Shape: [num_samples * batch_size, latent_dim]
        x_predict = decoder(z_samples)
        x_predict = x_predict.view(num_samples, batch_size, -1)  # Shape: [num_samples, batch_size, x_dim]
        
        # Compute log p(x|z) for x
        x_expanded = x.unsqueeze(0).expand(num_samples, -1, -1)  # Shape: [num_samples, batch_size, x_dim]
        
        # Compute log likelihoods and KL div
        if self.loss == "BCE":
            recon_log_prob = - nn.functional.binary_cross_entropy(x_predict, x_expanded, reduction='none').sum(-1)  # Shape: [num_samples, batch_size]
        else:
            recon_log_prob = - self.lambd_rec * 0.5 * nn.functional.mse_loss(x_predict, x_expanded, reduction='none').sum(-1)  # Shape: [num_samples, batch_size]
        
        KL_divergence = 0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1)  # [batch_size]

        # Mean recon_term over all samples
        elbo_samples = recon_log_prob.mean(0) + KL_divergence  # Shape: [batch_size]
        
        return elbo_samples

    def plot(self, iteration):
        n = 10
        digit_size = 28
        figure = np.zeros((digit_size * self.n_class, digit_size * n))

        for i in range(self.n_class):
            decoder = self.decoders[i]
            decoder.eval()  # Set decoder to evaluation mode
            for j in range(n):
                z_sample = torch.randn(1, self.latent_dim).to(self.device)
                with torch.no_grad(): 
                    x_decoded = decoder(z_sample)
                digit = x_decoded.view(1, 1, 28, 28).squeeze()
                figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit.detach().cpu().numpy()

        # Display the plot
        plt.figure(figsize=(20, 20))
        plt.imshow(figure, cmap='Greys_r')
        plt.axis('off')  

        # Save the figure 
        plt.savefig(f'{self.folder}/{self.file_path}/iter_{iteration}.png')

        plt.close()
            

    def solve(self):
        print("Start", flush = True)
        iteration = 0
        
        while iteration < self.n_iter:
            if iteration > 0 and iteration % 20 == 0:
                self.scheduler.step()
                print(f"Learning rate updated to {self.scheduler.get_last_lr()[0]} at iteration {iteration}", flush = True)
            # M-step
            self.maximization_step(iteration)
            # E-Step
            self.estimation_step(iteration)
            # Calculate test set accuracy
            self.test(iteration)

            # accuracy            
            accuracy = calc_accuracy(self.y, self.weights.T.cpu())
            accuracy_test = calc_accuracy(self.y_test, self.y_test_pred.T.cpu())
            print(f"Iteration {iteration}:", f'Clustering Accuracy {accuracy:.4f}', f'Clustering Accuracy Test {accuracy_test:.4f}', flush = True)

            if iteration % 30 == 0:
                self.plot(iteration)
                        
            iteration += 1
            
        return self.weights