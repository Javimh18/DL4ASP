import numpy as np 
from torch.optim import Adamax
from trainers import ViTTrainer
from vit import ViT
from torch.utils.data import DataLoader, RandomSampler
from dataset import SynthDataset
from torch.utils.data.dataloader import default_collate
import os

if __name__ == '__main__':
    log_spectrogram1 = np.load('./data/real/spectrograms/train/real_0_george_0.wav.npy')
    n_freq_bins, n_frames = log_spectrogram1.shape[1],log_spectrogram1.shape[0]
    
    model = ViT(img_size=(n_freq_bins, n_frames),
                patch_size=(32,8),
                in_chans=1,
                embed_dim=100,
                n_heads=10,
                n_layers=3,
                n_classes=1)
    
    # hyperparameters for training
    epochs = 30
    lr = 0.005
    weight_decay = 5e-7
    betas = (0.95, 0.999)
    batch_size = 128
    
    # instanciate datasets
    train_dataset = SynthDataset(path_to_dataset='./data/', extension=['.npy'], subset='train')
    val_dataset = SynthDataset(path_to_dataset='./data', extension=['.npy'], subset='val')
    
    # instanciate dataloaders
    train_dataloader = DataLoader(dataset=train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  num_workers=os.cpu_count(),
                                  collate_fn=default_collate)
    
    val_dataloader = DataLoader(dataset=val_dataset, 
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=os.cpu_count(),
                                collate_fn=default_collate)
    
    # Setting up for training
    trainables = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adamax(trainables, lr=lr, weight_decay=weight_decay, betas=betas)
    trainer = ViTTrainer(model,
                         optimizer,
                         train_dataloader,
                         val_dataloader,
                         epochs,
                         None)
    trainer.train()