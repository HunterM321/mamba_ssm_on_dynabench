import torch
import torch.nn as nn
from ..mamba_library import MambaTower

class MambaSequencePredictor(nn.Module):
    def __init__(self, input_dim, d_model, output_dim, n_layers, global_pool=False, do_norm=True, dropout_level=0, ssm_layer='mamba', **kwargs):
        """
        A sequence predictor model that wraps a MambaTower with linear layers around it.

        Parameters:
        ----------
        input_dim : int
            Dimensionality of input features.
        d_model : int
            Dimensionality of features inside the MambaTower.
        output_dim : int
            Dimensionality of the output (number of classes or regression target dimension).
        n_layers : int
            Number of MambaBlocks to stack in the MambaTower.
        global_pool : bool, optional (default=False)
            If True, the output of MambaTower will be pooled to (B, D) for sequence classification.
        do_norm : bool, optional (default=True)
            Whether to apply LayerNorm in each MambaBlock.
        dropout_level : float, optional (default=0)
            Dropout fraction in each MambaBlock.
        ssm_layer : str, optional (default='mamba')
            SSM layer type to use in each MambaBlock.
        **kwargs : dict
            Additional arguments for Mamba configuration in each block.
        """
        super(MambaSequencePredictor, self).__init__()
        
        # Initial linear layer to map input to d_model
        self.input_layer = nn.Linear(input_dim, d_model)
        
        # MambaTower as the core sequence model
        self.mamba_tower = MambaTower(d_model, n_layers, global_pool, do_norm, dropout_level, ssm_layer, **kwargs)
        
        # Output layer to map from d_model to output_dim
        if global_pool:
            self.output_layer = nn.Linear(d_model, output_dim)
        else:
            self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, x):
        """
        Forward pass for the sequence predictor.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor of shape (B, N, input_dim), where B is batch size, N is sequence length.

        Returns:
        ----------
        torch.Tensor
            Output tensor of shape (B, N, output_dim) if global_pool is False, otherwise (B, output_dim).
        """
        # Pass through initial linear layer
        x = self.input_layer(x)
        
        # Pass through MambaTower
        x = self.mamba_tower(x)
        
        # Pass through output layer
        x = self.output_layer(x)
        
        return x
