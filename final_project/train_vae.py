import numpy as np 
from torch.optim import Adamax
from trainers import VAETrainer
from vae import VAEAudio
from torch.utils.data import DataLoader
from dataset import VAEDataset
from torch.utils.data.dataloader import default_collate
import os

if __name__ == '__main__':
    log_spectrogram1 = np.load('./data/real/spectrograms/train/0_george_0.wav.npy')
    n_freq_bins, n_frames = log_spectrogram1.shape[1],log_spectrogram1.shape[0]
    model = VAEAudio(input_size=(1, n_freq_bins, n_frames),
                     latent_dim=128,
                     n_convLayers=5,
                     n_convChannels=[32, 64, 128, 256, 512],
                     filter_sizes=[4, 3, 3, 3, 3],
                     strides=[2, 2, 2, 2, (2,1)],
                     n_fcLayer=1, 
                     n_hidden_dims=[256])
    
    # hyperparameters for training
    epochs = 150
    lr = 0.005
    weight_decay = 5e-7
    betas = (0.95, 0.999)
    
    # instanciate datasets
    train_dataset = VAEDataset(path_to_dataset='./data/real/', extension=['.npy'], subset='train')
    val_dataset = VAEDataset(path_to_dataset='./data/real/', extension=['.npy'], subset='val')
    
    # instanciate dataloaders
    train_dataloader = DataLoader(dataset=train_dataset, 
                                  batch_size=128, 
                                  shuffle=True, 
                                  num_workers=os.cpu_count(),
                                  collate_fn=default_collate)
    
    val_dataloader = DataLoader(dataset=val_dataset, 
                                batch_size=128, 
                                shuffle=True, 
                                num_workers=os.cpu_count(),
                                collate_fn=default_collate)
    
    # Setting up for training
    trainables = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adamax(trainables, lr=lr, weight_decay=weight_decay, betas=betas)
    trainer = VAETrainer(model, 
                         optimizer, 
                         train_dataloader, 
                         val_dataloader, 
                         epochs, 
                         None)
    trainer.train()
    
    