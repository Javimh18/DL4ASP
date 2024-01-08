import torch
from preprocessing import MinMaxNormaliser
import librosa
from tqdm import tqdm
import numpy as np
def reshape(x):
        n_freqBand, n_contextWin = x.size(2), x.size(1)
        return x.view(-1, 1, n_freqBand, n_contextWin)  

class Synthesizer():
    """
    This class is in charge of synthesizing new data using the VAE. To do this
    we will forward the test data to the VAE, recover the produced spectrograms
    and convert them from the Cepstral domain into the Time Domain.
    """
    
    def __init__(self, vae, hop_length):
        self.vae = vae
        self.hop_length = hop_length
        self._min_max_normaliser = MinMaxNormaliser(0,1) # for deconversion from normalized log-spectrogram
                                                         # to unnormalized scale
    
    def get_signals_from_spectrograms(self, dataloader, device):   
        # set the model in evaluation mode
        self.vae.eval()
        signals_z_list = []
        with torch.no_grad():
            for (idx, data, (max_values, min_values)) in tqdm(dataloader):
                x = data.type('torch.FloatTensor').to(device)
                x = reshape(x)
                
                generated_spectrograms, signals, z, idx = self.synthesize(x, max_values, min_values, idx)
                signals_z_list.append((generated_spectrograms, signals, z, idx))
                
        return signals_z_list
                                                         
    def synthesize(self, spectrograms, min_values, max_values, idx):
        generated_spectrograms, _, _, z_s = self.vae(spectrograms)
        
        # convert latent output tensors to numpy arrays. They share the
        # same memory space, so be careful with modifications, if a value
        # is changed in the numpy array, the torch tensor will be affected by it
        out_generated_spectrograms_np = generated_spectrograms.detach()\
                                                              .squeeze()\
                                                              .cpu()\
                                                              .numpy()
        z_s_np = z_s.detach().cpu().numpy()
        idx_np = idx.detach().unsqueeze(-1).cpu().numpy()
        
        # B, C, freq_bins, n_frames -> B, freq_bins, n_frames, C
        generated_spectrograms = generated_spectrograms.permute(0,2,3,1)
        
        # convert spectrograms to signals
        signals = self.reconvert_spectrograms_to_audio(generated_spectrograms, min_values, max_values)
        return out_generated_spectrograms_np, signals, z_s_np, idx_np
    
    def reconvert_spectrograms_to_audio(self, spectrograms, min_values, max_values):
        # convert spectrogram and min_max_values to numpy ndarray
        spectrograms = spectrograms.cpu().numpy()
        min_value = min_values.cpu().numpy()
        max_value = max_values.cpu().numpy()
        
        signals = []
        for spectrogram, min_value, max_value in zip(spectrograms, min_value, max_value):
            # reshape the log spectrogram (squeezing the channel dimension)
            log_spectrogram = spectrogram[:,:,0]
            # appplying denormalization (min_max_values)
            denorm_log_spectrogram = self._min_max_normaliser.denormalise(
                log_spectrogram,
                min_value,
                max_value
            )
            # log_spectrogram -> linear spectrogram
            lin_spec = librosa.db_to_amplitude(denorm_log_spectrogram)
            # apply algo to convert from cepstral domain to time domain
            signal = librosa.istft(lin_spec, hop_length=self.hop_length)
            # add it to the list of putput signals
            signals.append(signal)

        signals = np.stack(signals)
        return signals
    
            


        
    