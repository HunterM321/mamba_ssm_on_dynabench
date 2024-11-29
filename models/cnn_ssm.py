from models.mamba_library import MambaTower

import torch
import torch.nn as nn

from typing import Tuple
from einops import rearrange

class CNN(nn.Module):
    """
        Simple 2D CNN model for grid data, adopted from Dynabench's implementation of the CNN module inside NeuralPDE

        Parameters
        ----------
        input_size : int
            Number of input channels.
        output_size : int
            Number of output channels.
        scaling : str
            Downsampling or upsampling.
        hidden_layers : int
            Number of hidden layers. Default is 1.
        hidden_channels : int
            Number of channels in each hidden layer. Default is 64.
        padding : int | str | Tuple[int]
            Padding size. If 'same', padding is calculated to keep the input size the same as the output size. Default is 'same'.
        padding_mode : str
            What value to pad with. Can be 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
        kernel_size : int
            Size of the kernel. Default is 3.
        activation : str
            Activation function to use. Can be one of `torch.nn activation functions <https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity>`_. Default is 'relu'.

    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 scaling: str,
                 hidden_layers: int = 1,
                 hidden_channels: int = 64,
                 padding: int | str | Tuple[int] = 'same',
                 padding_mode: str = 'circular',
                 kernel_size: int = 3,
                 activation: str = 'ReLU'):
        super().__init__()
        self.scaling = scaling
        self.input_layer = nn.Conv2d(input_size, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode)
        self.hidden_layers = nn.ModuleList([nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode) for _ in range(hidden_layers)])
        self.output_layer = nn.Conv2d(hidden_channels, output_size, kernel_size, padding=padding, padding_mode=padding_mode)

        self.activation = getattr(nn, activation)()

        # For downscaling and upscaling
        if self.scaling == 'down':
            self.scale_conv = nn.Conv2d(output_size, output_size, kernel_size=3, stride=3, padding=0)
        else:
            self.scale_conv = nn.ConvTranspose2d(output_size, output_size, kernel_size=3, stride=3, padding=0)

    def forward(self, x: torch.Tensor):
        """
            Forward pass of the model. Should not be called directly, instead call the model instance.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape (batch_size, input_size, height, width).

            Returns
            -------
            torch.Tensor
                Output tensor of shape (batch_size, output_size, height, width).
        """
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)

        x = self.scale_conv(x)

        return x
    
class MambaCNNMOL(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_layers,
                 hidden_channels):
        super().__init__()

        # CNN to downsample the input into lower spatial dimensions
        self.in_cnn = CNN(input_size=input_size,
                          output_size=input_size,
                          scaling='down',
                          hidden_layers=hidden_layers,
                          hidden_channels=hidden_channels)
        
        # CNN to upsample the SSMed input into the original spatial dimensions
        self.out_cnn = CNN(input_size=input_size,
                           output_size=output_size,
                           scaling='up',
                           hidden_layers=hidden_layers,
                           hidden_channels=hidden_channels)
        
        self.mamba = MambaTower(d_model=25,
                                n_layers=3,
                                ssm_layer='mamba')
    
    def forward(self, x):
        # Extract the feature dimension
        x = torch.squeeze(x, dim=2)

        B, T, H, W = x.shape
        
        x = self.in_cnn(x)
        x = rearrange(x, 'b c h w -> b c (h w)')
        x = self.mamba(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=5, w=5)
        x = self.out_cnn(x)

        # Add the feature dimension
        x = torch.unsqueeze(x, dim=2)

        return x
