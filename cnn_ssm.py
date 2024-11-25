from dynabench.dataset import DynabenchIterator, download_equation
from mamba_library import MambaTower

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Tuple
from tqdm import tqdm
from einops import rearrange

Apple_computer = True
if Apple_computer:
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

class CNN(nn.Module):
    """
        Simple 2D CNN model for grid data.

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
        B, T, H, W = x.shape
        
        x = self.in_cnn(x)
        x = rearrange(x, 'b c h w -> b c (h w)')
        x = self.mamba(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=5, w=5)
        x = self.out_cnn(x)

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
        B, T, H, W = x.shape
        
        x = self.in_cnn(x)
        x = rearrange(x, 'b c h w -> b c (h w)')
        x = self.mamba(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=5, w=5)
        x = self.out_cnn(x)

        return x

lookback = 3
rollout = 3

advection_train_iterator = DynabenchIterator(split="train",
                                             equation='advection',
                                             structure='grid',
                                             resolution='low',
                                             lookback=lookback,
                                             rollout=rollout)

train_loader = DataLoader(advection_train_iterator, batch_size=32, shuffle=True)

model = MambaCNNMOL(input_size=lookback,
                 output_size=rollout,
                 hidden_layers=3,
                 hidden_channels=9).to(device)
model.eval()

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(10):
    model.train()
    # Use tqdm for the outer loop to show epoch progress
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{10}", unit="batch") as pbar:
        for i, (x, y, p) in enumerate(train_loader):
            x, y = x[:, :, 0].float().to(device), y[:, :, 0].float().to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            
            # Update the progress bar with loss information
            pbar.set_postfix({"Loss": loss.item()})
            pbar.update(1)

advection_test_iterator = DynabenchIterator(split="test",
                                            equation='advection',
                                            structure='grid',
                                            resolution='low',
                                            lookback=3,
                                            rollout=3)

test_loader = DataLoader(advection_test_iterator, batch_size=32, shuffle=False)

loss_values = []
with tqdm(total=len(test_loader), desc="Testing", unit="batch") as pbar:
    for i, (x, y, p) in enumerate(test_loader):
        x, y = x[:, :, 0].float(), y[:, :, 0].float()
        y_pred = model(x, t_eval=range(17))
        loss = criterion(y_pred, y)
        
        # Append the loss to the list
        loss_values.append(loss.item())
        
        # Update the progress bar with loss information
        pbar.set_postfix({"Loss": loss.item()})
        pbar.update(1)

print(f"Mean Loss: {sum(loss_values) / len(loss_values)}")
