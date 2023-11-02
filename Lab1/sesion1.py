import os
import librosa as lr
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
from dl4asp_pr1 import plot_waveform, plot_melgram, plot_labels
from IPython.display import Audio, display

FEAT_SUBPATH = 'features/sr44100_win2048_hop511_mels64_nolog/'
AUDIO_SUBPATH = 'audio/validation/'
META_SUBPATH = 'metadata/validation/'
META_FILE = 'validation.tsv'

CLASSES = np.array(['Alarm_bell_ringing', 'Blender', 'Cat', 'Dishes', 'Dog', 
           'Electric_shaver_toothbrush', 'Frying', 'Running_water', 
           'Speech', 'Vacuum_cleaner'], dtype='object')


if __name__ == '__main__':
    
    # segment for this session
    segment = 'Y_URfKwl2kYU_130.000_140.000'
    
    
    # %% Waveforms and audio features
    # What is the value of the sample frequency (fs) # plots the waveform
    x, fs = plot_waveform(segment)
    print(f"The value of the sample frequency is: {fs}")
    # What is the duration of the audio file in seconds?
    print(f"The time in seconds of the audio file is: {x.size / fs}")
    
    # Mel spectrogram representation
    mel = plot_melgram(segment)
    print(f"Mel spectrogram info: {mel.shape}")
    
    # %% Event Annotations
    plot_labels(segment)
    
    