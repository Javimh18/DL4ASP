import torch
import numpy as np
from torch import nn
import librosa
from preprocessing import MinMaxNormaliser
from data_loaders import FSSDDataLoader

class Synthesizer():
    """
    This class is in charge of synthesizing new data using the VAE. To do this
    we will forward the test data to the VAE, recover the produced spectrograms
    and convert them from the Cepstral domain into the Time Domain.
    """
    
    def __init__(self, vae, ) -> None:
        pass