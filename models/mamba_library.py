# from mamba_ssm import Mamba, Mamba2
# from mamba_minimal.model import ModelArgs
use_real_mamba = False

if use_real_mamba:
    from mamba_ssm import Mamba, Mamba2
else:
    from models.mamba_minimal_model import ModelArgs, MambaBlock_simple, ResidualBlock
import os
import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np

class MambaBlock(nn.Module):
    """
    A single Mamba or Mamba2 block with added layer normalization and dropout.
    In particular, this block has all of the built-in Mamba configuration options.
    ~ 3 * expand * d_model^2 parameters, per block, approximately.

    Parameters:
    ----------
    d_model : int
        Input/output dimension.
    do_norm : bool, optional (default=True)
        Toggle for layer normalization.
    dropout_level : float, optional (default=0)
        Dropout fraction. Default is 0, indicating no dropout.
    ssm_layer : str, optional (default='mamba')
        SSM used. Should be 'mamba' or 'mamba2'. Raises ValueError if invalid option is provided.
    **kwargs : dict
        Additional keyword arguments for Mamba configuration. https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
    """
    def __init__(self, d_model, do_norm=True, dropout_level=0, ssm_layer='mamba', **kwargs):
        super().__init__()

        self.do_norm = do_norm
        self.dropout_level = dropout_level
        self.ssm_layer = ssm_layer
        if ssm_layer not in ['mamba', 'mamba2']:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support 'mamba' and 'mamba2'.")

        # Default configuration for Mamba (missing d_model since this is a positional argument)
        mamba_config = {
            'd_state': 16,
            'd_conv': 4,
            'expand': 2,
            'dt_rank': "auto",
            'dt_min': 0.001,
            'dt_max': 0.1,
            'dt_init': "random",
            'dt_scale': 1.0,
            'dt_init_floor': 1e-4,
            'conv_bias': True,
            'bias': False,
            'use_fast_path': True,
            'layer_idx': None,
            'device': None,
            'dtype': None,
        }
        # Default configuration for Mamba2 (missing d_model since this is a positional argument)
        mamba2_config = {
            'd_state': 64,
            'd_conv': 4,
            'conv_init': None,
            'expand': 2,
            'headdim': 128,
            'ngroups': 1,
            'A_init_range': (1,16),
            'dt_min': 0.001,
            'dt_max': 0.1,
            'dt_init_floor': 1e-4,
            'dt_limit': (0.0, float("inf")),
            'learnable_init_states': False,
            'acivation': "swish",
            'bias': False, 
            'conv_bias': True,
            # Fused kernel and sharding options
            'chunk_size': 256,
            'use_mem_eff_path': True,
            'layer_idx': None,
            'device': None,
            'dtype': None,
        }

        # Update default configuration with any provided keyword arguments (ignoring non-mamba KVs)
        if self.ssm_layer == 'mamba':
            mamba_config.update({k: v for k, v in kwargs.items() if k in mamba_config})
            # Initialize Mamba with updated configuration
            self.ssm = Mamba(d_model=d_model, **mamba_config)
        else:
            mamba2_config.update({k: v for k, v in kwargs.items() if k in mamba2_config})
            # Initialize Mamba2 with updated configuration
            self.ssm = Mamba2(d_model=d_model, **mamba2_config)  
               
        if self.do_norm:
            self.norm = nn.LayerNorm(d_model)
        if self.dropout_level > 0:
            self.dropout = nn.Dropout(self.dropout_level)

    def forward(self, x):
        x = self.ssm(x)
        if self.do_norm:
            x = self.norm(x)
        if self.dropout_level > 0:
            x = self.dropout(x)
        return x


class MambaTower(nn.Module):
    """
    A sequential or pooling block for multiple MambaBlocks with shared configurations.

    Parameters:
    ----------
    d_model : int
        Expected input/output dimension.
    n_layers : int
        Number of MambaBlocks to chain in sequence.
    global_pool : bool, optional (default=False)
        If True, the output will be pooled to dimension (B, D). Otherwise, it will be (B, N, D).
    do_norm : bool, optional (default=True)
        Toggle for layer normalization in each block.
    dropout_level : float, optional (default=0)
        Dropout fraction for each block.
    ssm_layer : str, optional (default='mamba')
        SSM used. Should be 'mamba' or 'mamba2'. Raises ValueError if invalid option is provided.
    **kwargs : dict
        Additional keyword arguments for Mamba configuration.
    """
    def __init__(self, d_model, n_layers, global_pool=False, do_norm=True, dropout_level=0, ssm_layer=None, **kwargs):
        super().__init__()

        self.args = ModelArgs(d_model=d_model,
                              n_layer=n_layers,
                              vocab_size=3)

        self.global_pool = global_pool
        if ssm_layer is None: ssm_layer = 'mamba'
        # Create MambaBlocks with shared configuration
        if use_real_mamba:
            self.blocks = nn.ModuleList([
                MambaBlock(d_model, do_norm=do_norm, dropout_level=dropout_level, ssm_layer=ssm_layer, **kwargs)
                for _ in range(n_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                ResidualBlock(args=self.args)
                for _ in range(n_layers)
            ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        if self.global_pool:
            x = torch.mean(x, dim=1)
        return x

class Mamba_ImgBlock(nn.Module):
    """
    Mamba_ImgBlock: Block making Mamba usuable on image classification.
                    Divides images into "sequences" of patches so images are compatible with Mamba.
                    Image dimensions (channel, length, width) are inferred on forward pass.
                    Note that the output is an embedding, NOT logits.          

    Shapes: `(B, C, d1, d2) --> (B, embed_dim)`
                    
    Patching algorithm:
    1. **Divide Image into Patches:** 
       The input image is divided into non-overlapping patches of size `(patch_size, patch_size)`, such that the input shape is `(B, C, h * patch_size, w * patch_size)`.

    2. **Flatten Patches:** 
       Each patch is flattened and channel-stacked into a 1D vector of size `(C * patch_size * patch_size)`. This is a single sequence item.

    3. **Sequence Formation:**
       The items are concatenated across the height and width of the image, forming a sequence with shape `(B, h * w, C * patch_size * patch_size)`.

    4. **Normalization across Patch Elements:**
       LayerNorm is applied across each item in the sequence (i.e. across one image)

    5. **Linear Mapping into `embed_dim`:**
       Each item is linearly mapped to an embedding of dimension `embed_dim`.

    6. **Normalization across Embeddings:**
       Another LayerNorm is applied across the embeddings.

    7. **MambaTower:**
       The sequence is passed through a `MambaTower` consisting of `n_layers` Mamba blocks to produce the final embeddings.

    Parameters:
    ----------
    patch_size : int
        Linear dimension of patch that images are divided into.
    embed_dim : int, optional (default=4)
        Embedding dimension. Unless expand is specified, this is 0.5x of dimension seen by SSM.
    n_layers : int, optional (default=1)
        Number of MambaBlocks to chain in sequence.
    do_norm : bool, optional (default=False)
        Toggle for layer normalization after each MambaBlock.
    dropout_level : float, optional (default=0)
        Dropout fraction for each MambaBlock.
    ssm_layer : str, optional (default='mamba')
        SSM used. Should be 'mamba' or 'mamba2'. Raises ValueError if invalid option is provided.
    **kwargs : dict
        Additional keyword arguments for Mamba configuration.

    Raises:
        ValueError if Module called on input whose (length, width) dimensions are not divisible by self.patch_size.
    """
    def __init__(self, patch_size=4, embed_dim=256, n_layers=1, do_norm=False, dropout_level=0, ssm_layer='None', **kwargs):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.dropout = dropout_level
        self.n_layers = n_layers
        self.do_norm = do_norm
        self.dropout = dropout_level

        # to be used to defined shape-dependent layers in forward()
        self.device = kwargs.get('device', 'cpu')

        # define shape-agnostic layers
        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)',
                            p1=self.patch_size, p2=self.patch_size)
        self.patch_layerNorm = None
        self.embed_proj = None
        self.embed_layerNorm = nn.LayerNorm(embed_dim)
        # global_pool is always True for classification - we do not predict an output sequence
        self.mamba = MambaTower(d_model=embed_dim, n_layers=n_layers, do_norm=do_norm, global_pool=True, dropout_level=self.dropout, ssm_layer=ssm_layer, **kwargs)

        self.func = nn.Identity()

    def forward(self, x):

        b, c, l1, l2 = x.size()
        patch_dim = c * self.patch_size * self.patch_size

        # Dynamically create model based on input shape - should happen only once
        if self.patch_layerNorm is None or self.embed_proj is None:

            self.patch_layerNorm = nn.LayerNorm(patch_dim).to(device=self.device)
            self.embed_proj = nn.Linear(patch_dim, self.embed_dim).to(device=self.device)

            self.feature_proj = nn.Sequential(self.rearrange, self.patch_layerNorm, self.embed_proj, self.embed_layerNorm, self.mamba).to(device=self.device)

        if l1 % self.patch_size != 0 or l2 % self.patch_size != 0:
             raise ValueError('Image dimensions are not divisible by patch size')

        return self.feature_proj(x)
