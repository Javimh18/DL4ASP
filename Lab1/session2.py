import os
import librosa as lr
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
from dl4asp_pr1 import plot_predictions, plot_labels_predictions
from IPython.display import Audio, display

FEAT_SUBPATH = 'features/sr44100_win2048_hop511_mels64_nolog/'
AUDIO_SUBPATH = 'audio/validation/'
META_SUBPATH = 'metadata/validation/'
META_FILE = 'validation.tsv'

CLASSES = np.array(['Alarm_bell_ringing', 'Blender', 'Cat', 'Dishes', 'Dog', 
           'Electric_shaver_toothbrush', 'Frying', 'Running_water', 
           'Speech', 'Vacuum_cleaner'], dtype='object')


if __name__ == '__main__':
    
    # segments for this session
    #segment = 'Y_URfKwl2kYU_130.000_140.000'
    #segment = 'Y_ZZpAj4rIRk_80.000_90.000'
    segment = 'Y0AA0R8gbxcI_30.000_40.000'
    
    # predictions file
    predfile = 'validation2019_predictions.tsv'
    
    plot_predictions(segment, predfile)
    
    plot_labels_predictions(segment, predfile)