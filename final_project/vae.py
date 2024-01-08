import torch
from torch import nn
import numpy as np 

class VAEAudio(nn.Module):
    '''
    Variational Autoencoder model. This model will implement the VAE for
    audio processing. It will take as input the audio features and it will
    reproduce them at the output by sampling from a latent space.
    
    :param input_size: (n_channel, n_freqBand, n_contextWin) It is assumed
                        to be a spectrogram of shape (n_freqBand, n_contextWin).
    :param latent_dim: the dimension of the latent vector.
    :param is_featExtract: Outputs z and mu if true, else z from the reparametri-
                            zation trick.
    '''
    def __init__(self, input_size=(1,64,256), # C,n_freq_bins,n_frames
                 latent_dim=128, 
                 n_convLayers=3, 
                 n_convChannels=[128, 64, 32], 
                 filter_sizes=[1, 4, 3], 
                 strides=[1, 2, 2],
                 n_fcLayer=1, 
                 n_hidden_dims=[512]):
        
        """
        Constructor class for VAE Audio.
        :param input_size: The mel spectrogram shape.
        :param latent_dim: the dimension of the latent vectors.
        :param n_convLayers: number of convlayers for the autoencoder .
        :param n_convChannel: Array with the channels of the autoencoder's encoder and
                decoder. They are inversed in order in the decoder.
        :param filter_sizes: Array with the number of filters of the autoencoder's encoder and
                decoder. They are inversed in order in the decoder.
        :param strides: Array with the number of filters of the autoencoder's encoder and
                decoder. They are inversed in order in the decoder.
        :param n_fcLayer: number of FC layers from the encoder output features.
        :param n_hidden_dims: The dimensions of the hidden layers from the FC block.
        """
        super(VAEAudio, self).__init__()
        
        self.input_size = input_size
        self.latent_dim = latent_dim
        
        self.channels_in, self.n_freq_band, self.n_context_win = input_size

        self.enc = self._enc_conv_layers(n_convLayers, 
                                   [self.channels_in] + n_convChannels,
                                   filter_sizes,
                                   strides)
        
        self.flattened_size, self.encoder_output_size = self._get_flat_size()
        self.enc_fc = self._fully_connected(n_fcLayer, [self.flattened_size] + n_hidden_dims)
        
        self.mu = self._fully_connected(1, [n_hidden_dims[-1], latent_dim])
        self.log_var = self._fully_connected(1, [n_hidden_dims[-1], latent_dim])
        
        self.dec_fc = self._fully_connected(n_fcLayer+1, [latent_dim, *n_hidden_dims[::-1], self.flattened_size])
        self.dec = self._dec_deconv_layers(n_convLayers, 
                                     n_convChannels[::-1] + [self.channels_in],
                                     filter_sizes[::-1],
                                     strides[::-1])
        
    def encode(self, x):
        """
        Takes the input and maps it into a latent space vector z
        """
        h = self.enc(x)
        h2 = self.enc_fc(h.view(-1, self.flattened_size))
        mu = self.mu(h2)
        logvar = self.log_var(h2)
        mu, logvar, z = self._get_latent(mu, logvar)
                
        return mu, logvar, z
        
    def decode(self, z):
        h = self.dec_fc(z)
        x_recon = self.dec(h.view(-1, *self.encoder_output_size))
        return x_recon
    
    def forward(self, x):
        mu, logvar, z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z
    
    def _get_flat_size(self):
        encoder_output = self.enc(torch.ones(1, *self.input_size))
        return int(np.prod(encoder_output.size()[1:])), encoder_output.size()[1:]
    
    def _get_latent(self, mu, logvar):
        mu, logvar, z = self._sampling_gaussian(mu, logvar)
        return mu, logvar, z

    @staticmethod
    def _sampling_gaussian(mu, logvar):
        
        # device initialization
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = "cpu"
        
        sigma = torch.sqrt(torch.exp(logvar))
        eps = torch.distributions.normal.Normal(0, 1).sample(sample_shape=sigma.size()).to(device)
        z = mu + sigma * eps  # reparameterization trick
        return mu, logvar, z

    @staticmethod
    def _enc_conv_layers(n_layer=3, n_channel=[1,8,16,32], filter_size=[1,3,3], stride=[1,2,2]):
        '''
        This function adds convolutional blocks to the encoder. In our case we are going to imple-
        ment a 1D conv, but it could also be performed with 2D convs, as the input is a mel-filter 
        bank spectrogram.
        :param n_layer: number of conv. layers
        :param n_channel: in/out number of channels for each layer. The first channel has the number
                        frequency bands of the input spectrograms.
        :param filter_size: filter size that slides in the x-axis.
        :param stride: stride for the x-axis.
        :returns: an object (nn.Sequential) that has the stacked layers for the encoder
        '''
        
        assert len(n_channel) == n_layer + 1, "This must fulfill: len(n_channel) = n_layer + 1"
        ast_msg = "The following must fulfill: len(filter_size) == len(stride) == n_layer"
        assert len(filter_size) == len(stride) == n_layer, ast_msg
        
        # we are going to build the layers by looping through the parameters 
        enc_conv_layers = []
        for i in range(n_layer):
            in_channel, out_channel = n_channel[i:i+2] # we take the pairs of input output channels
            
            enc_conv_layers += [
                    nn.Conv2d(in_channel, out_channel, filter_size[i], stride[i]),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU()
            ]
            
        return nn.Sequential(*enc_conv_layers)

    @staticmethod
    def _dec_deconv_layers(n_layer=3, n_channel=[32,16,8,1], filter_size=[3,3,1], stride=[2,2,1]):
        '''
        This function adds deconvolutional blocks to the decoder. To "reverse" the encoding process
        we will make use of the deconv 1-D layers, inflating the spatial resolution.
        :param n_layer: number of deconv. layers
        :param n_channel: in/out number of channels for each layer. It should have the reversed order
                        of the encoder layers. The last channel has the number
                        frequency bands of the input spectrograms.
        :param filter_size: filter size that slides in the x-axis. It should have the reversed order
                        of the encoder layers
        :param stride: stride for the x-axis. It should have the reversed order
                        of the encoder layers
        :returns: an object (nn.Sequential) that has the stacked layers for the encoder
        '''
        
        assert len(n_channel) == n_layer + 1, "This must fulfill: len(n_channel) = n_layer + 1"
        ast_msg = "The following must fulfill: len(filter_size) == len(stride) == n_layer"
        assert len(filter_size) == len(stride) == n_layer, ast_msg
        
        # We are going to build the layers looping through the parameters
        dec_deconv_layers = []
        for i in range(n_layer):
            in_channel, out_channel = n_channel[i:i+2] # we take the pairs of input output channels
            dec_deconv_layers += [
                nn.ConvTranspose2d(in_channel, out_channel, filter_size[i], stride[i]),
                nn.BatchNorm2d(out_channel),
                nn.ReLU()
            ]
            
        return nn.Sequential(*dec_deconv_layers)

    @staticmethod
    def _fully_connected(n_layer, hidden_dim):
        '''
        Construction of the FC block
        :param n_layer: number of fc. layers
        :param hidden_dim: hidden_dims of the fc block
        '''
        
        assert len(hidden_dim) == n_layer + 1, "This must fulfill: len(n_channel) = n_layer + 1"
        
        fc_layers = []
        for i in range(n_layer):
            fc_layers += [
                nn.Linear(hidden_dim[i], hidden_dim[i+1]),
                nn.BatchNorm1d(hidden_dim[i+1]),
                nn.ReLU()
            ]
            
            
        return nn.Sequential(*fc_layers)