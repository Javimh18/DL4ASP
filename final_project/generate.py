import torch
from data_loaders import FSSDDataLoader
from torch.utils.data.dataloader import default_collate

from vae import VAEAudio
from dataset import VAEDataset
from synthesizer import Synthesizer

import soundfile as sf
import numpy as np
import os
import math
from tqdm import tqdm

def save_signals(signal, save_path, sample_rate=22050):
    sf.write(save_path, signal, sample_rate)

if __name__ == '__main__':
    HOP_LENGTH = 256
    REAL_DATASET_PATH = './data/real/'
    SYNTH_DATASET_PATH = './data/synth/'
    BATCH_SIZE = 128
    TRAIN_PROPORTION = 0.8
    TYPES_OF_DATA = ['audio','spectrogram']
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
    
    # TODO: load this in a more "dynamic" way
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
    train_dataset = VAEDataset(path_to_dataset=REAL_DATASET_PATH, extension=['.npy'], subset='train')
    val_dataset = VAEDataset(path_to_dataset=REAL_DATASET_PATH, extension=['.npy'], subset='val')
    test_dataset = VAEDataset(path_to_dataset=REAL_DATASET_PATH, extension=['.npy'], subset='test')
    
    # instanciate dataloaders
    train_dataloader = FSSDDataLoader(dataset=train_dataset, 
                                  batch_size=BATCH_SIZE, 
                                  shuffle=True, 
                                  num_workers=os.cpu_count(),
                                  collate_fn=default_collate)
    
    val_dataloader = FSSDDataLoader(dataset=val_dataset, 
                                batch_size=BATCH_SIZE, 
                                shuffle=True, 
                                num_workers=os.cpu_count(),
                                collate_fn=default_collate)
    
    test_dataloader = FSSDDataLoader(dataset=test_dataset, 
                                batch_size=BATCH_SIZE, 
                                shuffle=True, 
                                num_workers=os.cpu_count(),
                                collate_fn=default_collate)
    
    dataloaders = [train_dataloader, val_dataloader, test_dataloader]
    
    # iterate through all the subsets to obtain the signals
    gen_spectrograms_data = []
    signals_data = []
    z_s_data = []
    idx_s_data = []
    subset_data = []
    for dl in dataloaders:
        print(f"Extracting the generated signals for the {dl.subset} subset.")
        signals_metadata = synth.get_signals_from_spectrograms(dl, device)
        
        # unpack metada from the subset we are processing
        for gen_spectrograms, signals, z_s, idx_s in signals_metadata:
            gen_spectrograms_data.append(gen_spectrograms)
            signals_data.append(signals)
            z_s_data.append(z_s)
            idx_s_data.append(idx_s)
            subset_data += [dl.subset] * len(gen_spectrograms) 
             
    # stacking all the information in numpy arrays with shape (3000, *)
    gen_spectrograms_data = np.vstack(gen_spectrograms_data)
    signals_data = np.vstack(signals_data)
    z_s_data = np.vstack(z_s_data)
    idx_s_data = np.vstack(idx_s_data)
    
    train_cutoff = math.floor(len(subset_data) * TRAIN_PROPORTION)
    val_cutoff = math.floor((len(subset_data) - train_cutoff) * TRAIN_PROPORTION)
    test_cutoff = len(subset_data) - (train_cutoff + val_cutoff)
    # assert train_cutoff + val_cutoff + test_cutoff == 3000 # small check
    
    # creating the splits for saving the data
    splits = ['train', 'val', 'test']
    splits_it= iter(splits)
    current_split = next(splits_it)
    
    # creating the directory where we are going to save the data
    if not os.path.exists(SYNTH_DATASET_PATH):
        os.makedirs(SYNTH_DATASET_PATH)
        for types in TYPES_OF_DATA:
            path_split = os.path.join(SYNTH_DATASET_PATH, types)
            os.makedirs(path_split)
            
        for split in splits:
            path_split = os.path.join(SYNTH_DATASET_PATH, types, split)
            os.makedirs(path_split)
            
    print(f"Saving the generated spectrograms and audio files under {SYNTH_DATASET_PATH}")
    random_idx = np.random.permutation(len(subset_data))
    counter = 1
    for r_idx in tqdm(random_idx):
        # get the number of the files
        if counter == train_cutoff:
            current_split = next(splits_it)
        elif counter == (val_cutoff + train_cutoff):
            current_split = next(splits_it)
            
        gen_spec = gen_spectrograms_data[r_idx]
        signal = signals_data[r_idx]
        idx = idx_s_data[r_idx][0]
        subset = subset_data[r_idx]
        
        file_name_npy = os.listdir(os.path.join(REAL_DATASET_PATH, 'spectrograms', subset))[idx]
        file_name_wav = file_name_npy.split(".")[0] + '.wav'
        
        # saving signal file
        synth_path = os.path.join(SYNTH_DATASET_PATH, 'audio')
        synth_path_wav = os.path.join(synth_path ,file_name_wav)
        save_signals(signal, synth_path_wav)
        
        # saving numpy spectrogram file
        synth_path = os.path.join(SYNTH_DATASET_PATH, 'spectrogram', current_split)
        synth_path_npy = os.path.join(synth_path, file_name_npy)
        np.save(synth_path_npy, gen_spec)
            
        counter += 1
        
        

        
        
    
    