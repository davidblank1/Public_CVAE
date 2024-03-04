# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:07:44 2024

@author: blank.136
"""
import torch.nn as nn
from torch.nn import functional as F
import torch
import os
from torch.optim import Adam

BATCH_SIZE = 30
X_DIM = 401
HIDDEN_DIM = 200
MID_DIM = 100
LATENT_DIM = 50
LR = 1e-3
EPOCHS = 10
NAME_OF_MODEL_FILE = 'CVAE_Species1.pt'

class Condition(nn.Module):
    def __init__(self):
        #super(Condition, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 50),
            nn.BatchNorm1d(50),
            nn.LeakyReLU(0.2))

    def forward(self, labels):
        x = F.one_hot(labels, num_classes=2).float().reshape(-1, 2)
        return self.fc(x)


class Reshape(nn.Module):
    def __init__(self, *shape):
        #super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(-1, *self.shape)


class Encoder(nn.Module):
    def __init__(self):
        #super(Encoder, self).__init__()
        self.fc_input = nn.Linear(X_DIM + 2, HIDDEN_DIM)
        self.fc_mean = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.fc_var = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, labels):
        h = torch.cat((x, labels), 1)
        h = self.leaky_relu(self.fc_input(h))
        mean = self.fc_mean(h)
        log_var = self.fc_var(h)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self):
        #super(Decoder, self).__init__()
        self.fc_hidden = nn.Linear(LATENT_DIM + 2, MID_DIM)
        self.fc_output = nn.Linear(MID_DIM, X_DIM)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, labels):
        h = torch.cat((x, labels), 1)
        h = self.leaky_relu(self.fc_hidden(h))
        x_hat = torch.sigmoid(self.fc_output(h))
        return x_hat


class Model(nn.Module):
    def __init__(self, encoder, decoder):
        #super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z

    def forward(self, x, labels):
        mean, log_var = self.encoder(x, labels)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decoder(z, labels)
        return x_hat, mean, log_var

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + kld

def training(loader):
    encoder = Encoder()
    decoder = Decoder()
    model = Model(encoder, decoder)

    if os.path.isfile(NAME_OF_MODEL_FILE):
        model = torch.load(NAME_OF_MODEL_FILE)
        model.eval()
    else:
        optimizer = Adam(model.parameters(), lr=LR)
        print("Start training VAE...")
        model.train()

        for epoch in range(EPOCHS):
            overall_loss = 0
            for imgs, labels in loader:
                x = imgs
                labels = labels.type(torch.int64)
                labels = F.one_hot(labels, num_classes=2).float().reshape(-1, 2)
                optimizer.zero_grad()
                x_hat, mean, log_var = model(x, labels)
                loss = loss_function(x, x_hat, mean, log_var)
                overall_loss += loss.item()
                loss.backward()
                optimizer.step()
            print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", loss.item())
        print("Finish!!")
        model.eval()
        torch.save(model, 'Trained_Model/' + NAME_OF_MODEL_FILE)
        
    return model
        