# DL4ASP - Sound Event Detection
# AUDIAS / Diego de Benito / 2020

import os
import librosa as lr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import ticker


# %% Environment settings

DATASET_PATH = './dataset_reduced'
FEAT_SUBPATH = 'features/sr44100_win2048_hop511_mels64_nolog/'
AUDIO_SUBPATH = 'audio/validation/'
META_SUBPATH = 'metadata/validation/'
META_FILE = 'validation.tsv'

CLASSES = np.array(['Alarm_bell_ringing', 'Blender', 'Cat', 'Dishes', 'Dog', 
           'Electric_shaver_toothbrush', 'Frying', 'Running_water', 
           'Speech', 'Vacuum_cleaner'], dtype='object')


# %% Display audio files and features

def plot_waveform(segment):
    path = os.path.join(DATASET_PATH, AUDIO_SUBPATH, segment+'.wav')
    x, fs = lr.load(path, sr=44100)
    x_len_seconds = x.size / fs
    
    plt.plot(x)
    plt.xlim(0, x.size)
    plt.xlabel('Time (seconds)')
    plt.grid()
    plt.xticks(fs*np.arange(1 + x_len_seconds), np.arange(1 + x_len_seconds))
    plt.title(segment)
    plt.show()
    
    return x, fs

def plot_melgram(segment):
    path = os.path.join(DATASET_PATH, FEAT_SUBPATH, segment+'.npy')

    mel = np.load(path)
    mel = np.log10(1 + np.abs(mel))
    mel_len_seconds = len(mel)/86
        
    plt.imshow(mel.T, origin='lower', aspect='auto', cmap='Blues')
    
    plt.ylabel('Mel filters')
    plt.xlabel('Time (seconds)')
    plt.xticks(len(mel)*np.arange(1 + np.floor(mel_len_seconds))/np.floor(mel_len_seconds), np.arange(1 + np.floor(mel_len_seconds)))
    plt.title(segment)
    plt.show()
    
    return mel
    

# %% Display ground truth labels and predictions
    
def plot_labels(segment, figsize=(8,6)):
    
    labelfile = os.path.join(DATASET_PATH, META_SUBPATH, META_FILE)  
    label_tsv = pd.read_table(labelfile, sep='\s+')
        
    segment_labels = label_tsv.loc[label_tsv.filename==segment+'.wav']
    
    fig, ax = plt.subplots(figsize=figsize)
    for _, l in segment_labels.iterrows():
        on = l["onset"]
        off = l["offset"]
        ev = np.where(CLASSES == l["event_label"])[0][0]
        
        ax.broken_barh([(on, off-on)], (ev-0.5, 1), color='C'+str(ev), alpha=0.6)
        
    ax.set_xlim([0,10])
    ax.set_ylim([-0.5,9.5])
    ax.set_xlabel('Time (seconds)')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.grid(True)
    ax.set_yticks(range(10))
    ax.set_yticklabels(CLASSES)
    ax.invert_yaxis()
    
    ax.set_title(segment +' (ground truth)')
    plt.show()



def plot_predictions(segment, predfile):
      
    pred_tsv = pd.read_table(predfile, sep='\s+')
        
    segment_preds = pred_tsv.loc[pred_tsv.filename==segment+'.wav']
    
    fig, ax = plt.subplots()
    for _, p in segment_preds.iterrows():
        on = p["onset"]
        off = p["offset"]
        ev = np.where(CLASSES == p["event_label"])[0][0]
        
        ax.broken_barh([(on, off-on)], [ev-0.25, 0.5], color='C'+str(ev))
        
    ax.set_xlim([0,10])
    ax.set_ylim([-0.5,9.5])
    ax.set_xlabel('Time (seconds)')
    ax.grid(True)
    ax.set_yticks(range(10))
    ax.set_yticklabels(CLASSES)
    ax.invert_yaxis()
    
    ax.set_title(segment +' (predictions)')


def plot_labels_predictions(segment, predfile):
    
    labelfile = os.path.join(DATASET_PATH, META_SUBPATH, META_FILE)  
    label_tsv = pd.read_table(labelfile, sep='\s+')
    pred_tsv = pd.read_table(predfile, sep='\s+')
        
    segment_preds = pred_tsv.loc[pred_tsv.filename==segment+'.wav']
    segment_labels = label_tsv.loc[label_tsv.filename==segment+'.wav']
    
    fig, ax = plt.subplots()
    
    for _, l in segment_labels.iterrows():
        on = l["onset"]
        off = l["offset"]
        ev = np.where(CLASSES == l["event_label"])[0][0]
        
        ax.broken_barh([(on, off-on)], (ev-0.5, 1), color='C'+str(ev), alpha=0.6)
        
    for _, p in segment_preds.iterrows():
        on = p["onset"]
        off = p["offset"]
        ev = np.where(CLASSES == p["event_label"])[0][0]
        
        ax.broken_barh([(on, off-on)], [ev-0.25, 0.5], color='C'+str(ev))
        
    ax.set_xlim([0,10])
    ax.set_ylim([-0.5,9.5])
    ax.set_xlabel('Time (seconds)')
    ax.grid(True)
    ax.set_yticks(range(10))
    ax.set_yticklabels(CLASSES)
    ax.invert_yaxis()
    
    ax.set_title(segment +' (ground truth + predictions)')



