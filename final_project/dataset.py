import os
from pathlib import Path
from typing import Any
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import numpy as np
import pickle

class ToTensor(object):
    """
    convert ndarrays in sample to tensors.
    """
    def __call__(self, log_spectrogram):
        return torch.from_numpy(log_spectrogram)

class VAEDataset(Dataset):
    def __init__(self, path_to_dataset, extension=['wav', 'mp3', 'npy', 'pth'], subset=None, transform=ToTensor()):
        super().__init__()
        
        self.path_to_dataset = path_to_dataset
        self.extension = extension
        self.subset = subset
        self.file_names = []
        self.transform = transform
        self.data_type = 'spectrograms'
        
        # load different splits from the original dataset
        if self.subset == 'train':
            train_path = os.path.join(path_to_dataset, self.data_type, self.subset)
            self.file_names = os.listdir(train_path)
        elif self.subset == 'val':
            val_path = os.path.join(path_to_dataset, self.data_type, self.subset)
            self.file_names = os.listdir(val_path)
        elif self.subset == 'test':
            test_path = os.path.join(path_to_dataset, self.data_type, self.subset)
            self.file_names = os.listdir(test_path)
            
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # get the file name of the spectrogram to load
        file_name = os.path.join(self.path_to_dataset, 
                                 self.data_type,
                                 self.subset,
                                 self.file_names[idx])
        
        # load max_min dictionary for the spectrogram (for denormalizing purposes in post processing)
        path_to_min_max_dict = os.path.join(self.path_to_dataset, 'min_max_values.pkl')
        with open(path_to_min_max_dict, "rb") as f:
            min_max_values = pickle.load(f)
            min_max_for_spectrogram = min_max_values[self.file_names[idx]]
            max_value, min_value = torch.tensor(min_max_for_spectrogram['max']),\
                                   torch.tensor(min_max_for_spectrogram['min'])
        
        # load the spectrogram
        log_spectrogram = np.load(file_name)
        if self.transform: 
            log_spectrogram = ToTensor()(log_spectrogram)
        
        return idx, log_spectrogram, (max_value, min_value)
    
class SynthDataset(Dataset):
    def __init__(self, path_to_dataset, extension=['wav', 'mp3', 'npy', 'pth'], type_of_data = 'spectrograms',subset=None, transform=ToTensor()) -> None:
        super().__init__()         
    
        self.path_to_dataset = path_to_dataset
        self.extension = extension
        self.subset = subset
        self.file_names = []
        self.class_file_names = []
        self.labels = []
        self.transform = transform   
        self.type_of_data = type_of_data
        
        self.classes = ['real', 'synth'] # real -> 0, synth -> 1
            
        
        for cls in self.classes:
            label = 0 if cls == 'real' else 1
            if self.subset == 'train':
                train_path = os.path.join(path_to_dataset, cls, self.type_of_data, self.subset)
                file_names = os.listdir(train_path)
                self.file_names += file_names

                # Put the corresponding label
                self.labels += [label] * len(self.file_names)
                self.class_file_names += [cls] * len(self.file_names)
                    
            elif self.subset == 'val':
                val_path = os.path.join(path_to_dataset, cls, self.type_of_data, self.subset)
                file_names = os.listdir(val_path)
                self.file_names += file_names
                
                # Put the corresponding label
                self.labels += [label] * len(self.file_names)
                self.class_file_names += [cls] * len(self.file_names)
                    
            elif self.subset == 'test':
                test_path = os.path.join(path_to_dataset, cls, self.type_of_data, self.subset)
                file_names = os.listdir(test_path)
                self.file_names += file_names
                
                # Put the corresponding label
                self.labels += [label] * len(self.file_names)
                self.class_file_names += [cls] * len(self.file_names)
                
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        file_name = os.path.join(self.path_to_dataset, 
                                 self.class_file_names[idx],
                                 self.type_of_data,
                                 self.subset,
                                 self.file_names[idx])
        
        log_spectrogram = np.load(file_name)
        y_true = self.labels[idx]
        
        if self.transform: 
            log_spectrogram = ToTensor()(log_spectrogram)
            y_true = torch.tensor(y_true).long()
        
        return idx, (log_spectrogram, y_true)
                
        
            
            
        