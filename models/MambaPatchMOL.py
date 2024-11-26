from einops.layers.torch import Rearrange
import torch
import torch.nn as nn

from .mamba_library import MambaTower  # Relative import

class MambaPatchMOL(nn.Module):
    def __init__(self, patch_size, d_model, n_layers, time_handling="keep", **mamba_kwargs):
        """
        Args:
            patch_size (int): Size of each patch (assumes square patches).
            d_model (int): Dimensionality of the MambaTower.
            n_layers (int): Number of layers in the MambaTower.
            time_handling (str): How to handle the time dimension. Options:
                - "keep": Keep the time dimension.
                - "last": Take the last timestep.
                - "poolmean": Apply mean pooling over the time dimension.
                - "poolmax": Apply max pooling over the time dimension.
            **mamba_kwargs: Additional keyword arguments for MambaTower.
        """
        super().__init__()
        assert time_handling in {"keep", "last", "poolmean", "poolmax"}, \
            "time_handling must be one of 'keep', 'last', 'poolmean', or 'poolmax'"
        assert patch_size > 0, "patch_size must be >0"
        
        self.patch_size = patch_size
        self.patch_dim = 0
        self.patch_num = 0 # to be defined in forward pass
        self.d_model = d_model
        self.time_handling = time_handling
        self.n_layers = n_layers
        # to be used to defined shape-dependent layers in forward()
        self.device = mamba_kwargs.get('device', 'cuda')
        self.mamba_kwargs = mamba_kwargs

        ## Layers - to be defined from first sample
        self.patch_in = None
        self.linear_in = None
        # self.mamba_towers = []
        self.mamba_towers = None
        self.linear_out = None
        self.patch_out = None


    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, time, height, width, features).
        Returns:
            torch.Tensor: Processed tensor with shape depending on time_handling:
                - "keep": (batch, time, num_patches, d_model)
                - "last": (batch, num_patches, d_model)
                - "poolmean": (batch, num_patches, d_model)
                - "poolmax": (batch, num_patches, d_model)
        """
        device = self.device
        B, T, *U = x.shape # U is either [F, H, W] for grid or [N, F] for cloud

        # generate layers if this is first sample
        if self.patch_in is None:

            # grid data: (B, T, F, H, W)
            if len(U) == 3: 
                B, T, F, H, W = x.shape
                if H % self.patch_size != 0 or H % self.patch_size != 0:
                    raise ValueError('Grid dimensions are not divisible by patch size')
                
                self.patch_dim = self.patch_size**2 * F
                self.patch_num = H * W // self.patch_size**2

                # self.patch_in = Rearrange('b t f (h1 p1) (h2 p2) -> b (h1 h2) t (f p1 p2)', p1=self.patch_size, p2=self.patch_size).to(device=self.device)
                # self.patch_out = Rearrange('b (h1 h2) t (f p1 p2) -> b t f (h1 p1) (h2 p2)', h1=H//self.patch_size, h2=W//self.patch_size, p1=self.patch_size, p2=self.patch_size).to(device=self.device)
                self.patch_in = Rearrange('b t f (h1 p1) (h2 p2) -> (b h1 h2) t (f p1 p2)', p1=self.patch_size, p2=self.patch_size).to(device=self.device)
                self.patch_out = Rearrange('(b h1 h2) t (f p1 p2) -> b t f (h1 p1) (h2 p2)', h1=H//self.patch_size, h2=W//self.patch_size, p1=self.patch_size, p2=self.patch_size).to(device=self.device)

            # cloud data: (B, T, N, F)
            elif len(U) == 2:
                B, T, N, F = x.shape
                if N % self.patch_size != 0:
                    raise ValueError('Cloud dimension are not divisible by patch size')

                self.patch_dim = self.patch_size * F
                self.patch_num = N // self.patch_size

                # self.patch_in = Rearrange('b t (h1 p1) f -> b h1 t (f p1)', p1=self.patch_size).to(device=self.device)
                # self.patch_out = Rearrange('b h1 t (f p1) -> b t (h1 p1) f)', p1=self.patch_size, p2=self.patch_size).to(device=self.device)
                self.patch_in = Rearrange('b t (h1 p1) f -> (b h1) t (f p1)', p1=self.patch_size).to(device=self.device)
                self.patch_out = Rearrange('(b h1) t (f p1) -> b t (h1 p1) f)', p1=self.patch_size, p2=self.patch_size).to(device=self.device)

            self.linear_in = nn.Linear(self.patch_dim, self.d_model).to(device=self.device)

            # self.mamba_towers = nn.ModuleList([MambaTower(d_model=self.d_model,n_layers=self.n_layers, global_pool=False, **self.mamba_kwargs).to(device=self.device) for _ in range(self.patch_num)])
            self.mamba_towers = MambaTower(d_model=self.d_model,n_layers=self.n_layers, global_pool=False, **self.mamba_kwargs).to(device=self.device)

            self.linear_out = nn.Linear(self.d_model, self.patch_dim).to(device=self.device)

            # self.patch_out already defined
        # print(f"dim: {x.shape}")
        x = self.patch_in(x)
        # print(f"after patch dim: {x.shape}")
        x = self.linear_in(x)
        # print(f"after linear dim: {x.shape}")

        y = self.mamba_towers(x)
        # splits = torch.chunk(x, chunks=self.patch_num, dim=1) # 1 is patch dim after rearrange
        # # print(f"after split dim (of one tensor): {splits[0].squeeze(dim=1).shape}")
        # split_ys = [mamba(x_i.squeeze(dim=1)) for mamba, x_i in zip(self.mamba_towers, splits)] # squeeze to remove patch_dim
        # # print(f"after split dim + mamba (of one tensor): {split_ys[0].squeeze(dim=1).shape}")
        # y = torch.stack(split_ys, dim=1)
        # print(f"y dim after concatenating along patch dim: {y.shape}")
        y = self.linear_out(y)
        # print(f"after linear out: {y.shape}")
        y = self.patch_out(y)
        # print(f"after final out: {y.shape}")

        if self.time_handling == 'last':
            y = y[:,[-1],:]
        elif self.time_handling == 'poolmax':
            y = torch.max(y, dim=1, keepdim=True)
        elif self.time_handling == 'poolmean':
            y = torch.mean(y, dim=1, keepdim=True)
        return y