import torch
import numpy as np 
import torch.functional as F
import torch.nn as nn
from DL4ASP.Practica.final_project.trainers import VAETrainer
from vae import VAEAudio
from data_loaders import FSSDDataLoader
from dataset import VAEDataset

if __name__ == '__main__':
    
    x = torch.rand(128)
    n_layer = 2
    hidden_dim = [128,512,30240]
    
    fc_layers = []
    for i in range(n_layer):
        layer = [nn.Linear(hidden_dim[i], hidden_dim[i+1])]
        layer.append(
            nn.BatchNorm1d(hidden_dim[i+1])
        )
        layer.append(
            nn.ReLU()
        )
    fc_layers += layer
            
    layer = nn.Sequential(*fc_layers)
    
    print(layer(x))