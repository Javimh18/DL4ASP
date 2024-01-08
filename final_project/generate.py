import torch
from data_loaders import FSSDDataLoader
from torch.utils.data.dataloader import default_collate

from vae import VAEAudio
from dataset import VAEDataset
from synthesizer import Synthesizer

import soundfile as sf
import numpy as np
import os

def save_signals(signals, save_dir, sample_rate=22050):
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, str(i) + ".wav")
        sf.write(save_path, signal, sample_rate)

if __name__ == '__main__':
    HOP_LENGTH = 256
    
    log_spectrogram1 = np.load('./data/real/spectrograms/train/0_george_0.wav.npy')
    n_freq_bins, n_frames = log_spectrogram1.shape[1],log_spectrogram1.shape[0]
    
    vae_model = VAEAudio(input_size=(1, n_freq_bins, n_frames),
                     latent_dim=128,
                     n_convLayers=5,
                     n_convChannels=[32, 64, 128, 256, 512],
                     filter_sizes=[4, 3, 3, 3, 3],
                     strides=[2, 2, 2, 2, (2,1)],
                     n_fcLayer=1, 
                     n_hidden_dims=[256])
    
    
    vae_model.load_state_dict(torch.load('models/vae_best_150.pth'))
    
    synth = Synthesizer(
        vae_model,
        HOP_LENGTH
    )
    
    device = ''
    # device initialization
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = "cpu"

    # instanciate datasets
    train_dataset = VAEDataset(path_to_dataset='./data/real/', extension=['.npy'], subset='train')
    val_dataset = VAEDataset(path_to_dataset='./data/real/', extension=['.npy'], subset='val')
    test_dataset = VAEDataset(path_to_dataset='./data/real/', extension=['.npy'], subset='test')
    
    # instanciate dataloaders
    train_dataloader = FSSDDataLoader(dataset=train_dataset, 
                                  batch_size=128, 
                                  shuffle=True, 
                                  num_workers=os.cpu_count(),
                                  collate_fn=default_collate)
    
    val_dataloader = FSSDDataLoader(dataset=val_dataset, 
                                batch_size=128, 
                                shuffle=True, 
                                num_workers=os.cpu_count(),
                                collate_fn=default_collate)
    
    test_dataloader = FSSDDataLoader(dataset=test_dataset, 
                                batch_size=128, 
                                shuffle=True, 
                                num_workers=os.cpu_count(),
                                collate_fn=default_collate)
    
    dataloaders = [train_dataloader, val_dataloader, test_dataloader]
    
    # iterate through all the subsets to obtain the signals
    all_signals = {}
    signals_data = np.array([])
    z_s_data = np.array([])
    idx_s_data = np.array([])
    for dl in dataloaders:
        print(f"Extrating the generated signals for the {dl.subset} subset.")
        signals_metadata = synth.get_signals_from_spectrograms(dl, device)
        
        # unpack metada from the subset we are processing
        for signals, z_s, idx_s in signals_metadata:
            signals_data += signals
            z_s_data += z_s
            idx_s_data += idx_s
        
    np.array(signals_data).shape
        
        
    
    