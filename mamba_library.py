# from mamba_ssm import Mamba, Mamba2
# from model import Mamba, ModelArgs
from __future__ import annotations
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

        ssm_args = ModelArgs(
            d_model=d_model,
            n_layer=1,
            vocab_size=40000
        )

        self.ssm = Mamba(args=ssm_args)

        # # Update default configuration with any provided keyword arguments (ignoring non-mamba KVs)
        # if self.ssm_layer == 'mamba':
        #     mamba_config.update({k: v for k, v in kwargs.items() if k in mamba_config})
        #     # Initialize Mamba with updated configuration
        #     self.ssm = Mamba(d_model=d_model, **mamba_config)
        # else:
        #     mamba2_config.update({k: v for k, v in kwargs.items() if k in mamba2_config})
        #     # Initialize Mamba2 with updated configuration
        #     self.ssm = Mamba2(d_model=d_model, **mamba2_config)  
               
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

        self.global_pool = global_pool
        if ssm_layer is None: ssm_layer = 'mamba'
        # Create MambaBlocks with shared configuration
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, do_norm=do_norm, dropout_level=dropout_level, ssm_layer=ssm_layer, **kwargs)
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
    def __init__(self, patch_size=4, embed_dim=256, n_layers=1, do_norm=False, dropout_level=0, ssm_layer='mamba', **kwargs):
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







"""Simple, minimal implementation of Mamba in one file of PyTorch.

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""
# from __future__ import annotations
import math
import json
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum


@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        super().__init__()
        self.args = args
        
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)

        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Tie output projection to embedding weights.
                                                     # See "Weight Tying" paper


    def forward(self, x):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            logits: shape (b, l, vocab_size)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """
        # x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        # logits = self.lm_head(x)

        return x

    
    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        """Load pretrained weights from HuggingFace into model.
    
        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'
                            
        Returns:
            model: Mamba model with weights loaded
    
        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file
        
        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))
        
        
        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
        
        config_data = load_config_hf(pretrained_model_name)
        args = ModelArgs(
            d_model=config_data['d_model'],
            n_layer=config_data['n_layer'],
            vocab_size=config_data['vocab_size']
        )
        model = Mamba(args)
        
        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict)
        
        return model


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.args = args
        self.mixer = MambaBlock_1(args)
        self.norm = RMSNorm(args.d_model)
        

    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297
            
            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
            
        """
        output = self.mixer(self.norm(x)) + x

        return output
            

class MambaBlock_1(nn.Module):
    def __init__(self, args: ModelArgs):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)
        

    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
    
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)
        
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (b, l, d) = x.shape
        
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x)

        y = self.ssm(x)
        
        y = y * F.silu(res)
        
        output = self.out_proj(y)

        return output

    
    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        
        return y

    
    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
    
        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)
    
        Returns:
            output: shape (b, l, d_in)
    
        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
            
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]
        
        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        
        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []    
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
        
        y = y + u * D
    
        return y


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
