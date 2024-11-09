import argparse

#### USAGE ###
# Intended to easily allow configuration of MODELS and DATASETs from bash.

# Use the below methods to instantiate the required parser.

# Have either your main parser object or a subparser inherit that
# parser's args using the @parent argument in the constructor.

# I suggest using subparsers so that these config options
# are only parsed for when the object using them is selected.

# -------------------------- DATASETS -------------------------- #

import argparse

### Advection equation
def create_eqn_advection_parser():
    advection_parser = argparse.ArgumentParser(description='Configuration options for Advection equation dataset')
    advection_parser.add_argument('--velocity', type=float, default=1.0, help='Advection velocity')
    advection_parser.add_argument('--domain', type=float, nargs=2, default=[0.0, 1.0], help='Spatial domain as [start, end]')
    advection_parser.add_argument('--grid_points', type=int, default=100, help='Number of grid points in the spatial domain')
    advection_parser.add_argument('--time_steps', type=int, default=100, help='Number of time steps for simulation')
    return advection_parser

### Burgers equation
def create_eqn_burgers_parser():
    burgers_parser = argparse.ArgumentParser(description='Configuration options for Burgers equation dataset')
    burgers_parser.add_argument('--viscosity', type=float, default=0.1, help='Viscosity parameter')
    burgers_parser.add_argument('--domain', type=float, nargs=2, default=[0.0, 1.0], help='Spatial domain as [start, end]')
    burgers_parser.add_argument('--grid_points', type=int, default=100, help='Number of grid points in the spatial domain')
    burgers_parser.add_argument('--time_steps', type=int, default=100, help='Number of time steps for simulation')
    burgers_parser.add_argument('--initial_condition', type=str, default='sin', help='Initial condition (e.g., "sin", "gaussian")')
    return burgers_parser

### Gas dynamics equation
def create_eqn_gas_dynamics_parser():
    gas_dynamics_parser = argparse.ArgumentParser(description='Configuration options for Gas dynamics equation dataset')
    gas_dynamics_parser.add_argument('--gamma', type=float, default=1.4, help='Ratio of specific heats (adiabatic index)')
    gas_dynamics_parser.add_argument('--domain', type=float, nargs=2, default=[0.0, 1.0], help='Spatial domain as [start, end]')
    gas_dynamics_parser.add_argument('--grid_points', type=int, default=100, help='Number of grid points in the spatial domain')
    gas_dynamics_parser.add_argument('--time_steps', type=int, default=100, help='Number of time steps for simulation')
    gas_dynamics_parser.add_argument('--initial_density', type=float, default=1.0, help='Initial density')
    gas_dynamics_parser.add_argument('--initial_velocity', type=float, default=0.0, help='Initial velocity')
    return gas_dynamics_parser

### Kuramoto-Sivashinsky
def create_eqn_kuramoto_sivashinsky_parser():
    ks_parser = argparse.ArgumentParser(description='Configuration options for Kuramoto-Sivashinsky equation dataset')
    ks_parser.add_argument('--domain', type=float, nargs=2, default=[0.0, 1.0], help='Spatial domain as [start, end]')
    ks_parser.add_argument('--grid_points', type=int, default=100, help='Number of grid points in the spatial domain')
    ks_parser.add_argument('--time_steps', type=int, default=100, help='Number of time steps for simulation')
    ks_parser.add_argument('--initial_condition', type=str, default='random', help='Initial condition (e.g., "random", "sin")')
    return ks_parser

### Reaction-Diffusion
def create_eqn_reaction_diffusion_parser():
    rd_parser = argparse.ArgumentParser(description='Configuration options for Reaction-Diffusion equation dataset')
    rd_parser.add_argument('--diffusion_u', type=float, default=0.16, help='Diffusion coefficient for species U')
    rd_parser.add_argument('--diffusion_v', type=float, default=0.08, help='Diffusion coefficient for species V')
    rd_parser.add_argument('--domain', type=float, nargs=2, default=[0.0, 1.0], help='Spatial domain as [start, end]')
    rd_parser.add_argument('--grid_points', type=int, default=100, help='Number of grid points in the spatial domain')
    rd_parser.add_argument('--time_steps', type=int, default=100, help='Number of time steps for simulation')
    rd_parser.add_argument('--initial_condition', type=str, default='random', help='Initial condition for species concentrations')
    return rd_parser

### Wave equation
def create_eqn_wave_parser():
    wave_parser = argparse.ArgumentParser(description='Configuration options for Wave equation dataset')
    wave_parser.add_argument('--speed', type=float, default=1.0, help='Wave propagation speed')
    wave_parser.add_argument('--domain', type=float, nargs=2, default=[0.0, 1.0], help='Spatial domain as [start, end]')
    wave_parser.add_argument('--grid_points', type=int, default=100, help='Number of grid points in the spatial domain')
    wave_parser.add_argument('--time_steps', type=int, default=100, help='Number of time steps for simulation')
    wave_parser.add_argument('--initial_displacement', type=str, default='gaussian', help='Initial displacement condition (e.g., "gaussian", "sin")')
    wave_parser.add_argument('--initial_velocity', type=float, default=0.0, help='Initial velocity condition')
    return wave_parser

# --------------------------- MODELS --------------------------- #

def create_Mamba_parser():
    # mamba_ssm.Mamba --------------------
    # Purely config for current Mamba implementation.
    # For ssm_mamba.models.mixer_seq_simple.MambaLMHeadModel, this could be used with vars() to get 'ssm_cfg'
    # Most if not all of my models will implement this.
    # Note that this set of arguments is the union of config options for Mamba and Mamba2. Given arguments 
    mamba_parser = argparse.ArgumentParser(description='ssm_mamba model configurations - these are available for every model setup which uses mamba', add_help=False)

    # Common Mamba configurations
    mamba_parser.add_argument('--d_state', type=int, default=16, help='State space dimension for Mamba models')
    mamba_parser.add_argument('--d_conv', type=int, default=4, help='Local convolution kernel width preceding diagonal SSM for Mamba models')
    mamba_parser.add_argument('--expand', type=int, default=2, help='Linear expansion factor applied to input dimension, preceding SSM layer for Mamba models')
    mamba_parser.add_argument('--dt_rank', type=str, default='auto', help='Rank of the state transition matrix for Mamba models')
    mamba_parser.add_argument('--dt_min', type=float, default=0.001, help='Minimum time step for Mamba models')
    mamba_parser.add_argument('--dt_max', type=float, default=0.1, help='Maximum time step for Mamba models')
    mamba_parser.add_argument('--dt_init', type=str, default='random', help='Initialization method for time step in Mamba models')
    mamba_parser.add_argument('--dt_scale', type=float, default=1.0, help='Scale for time step in Mamba models')
    mamba_parser.add_argument('--dt_init_floor', type=float, default=1e-4, help='Initialization floor for time step in Mamba models')
    mamba_parser.add_argument('--conv_bias', action='store_true', default=True, help='Use bias in convolutional layers for Mamba models')
    mamba_parser.add_argument('--bias', action='store_true', default=False, help='Use bias in linear layers for Mamba models')
    mamba_parser.add_argument('--use_fast_path', action='store_true', default=True, help='Use fast path optimizations for Mamba models')
    mamba_parser.add_argument('--layer_idx', type=int, default=None, help='Layer index for Mamba models')
    mamba_parser.add_argument('--device', type=str, default=None, help='Device for Mamba models (e.g., "cpu" or "cuda")')
    mamba_parser.add_argument('--dtype', type=str, default=None, help='Data type for Mamba models')    

    return mamba_parser

def create_MambaTower_parser():
    # ------- ssm_library.mamba_library.MambaTower --------------------
    # Inherits Mamba() config, plus a few additions for MambaTower.

    mamba_parser = create_Mamba_parser()

    mambaTower_parser = argparse.ArgumentParser(description='ssm_mamba model configurations - these are available for every model setup which uses mamba', add_help=False, parents=[mamba_parser])

    ## My custom options for MambaTower
    mambaTower_parser.add_argument('--do_norm', action='store_true', default=True, help='CUSTOM: Apply layer normalization after each MambaBlock in Mamba models')
    mambaTower_parser.add_argument('--dropout_level', type=float, default=0, help='CUSTOM: Dropout level for each MambaBlock in Mamba models')
    mambaTower_parser.add_argument('--n_layers', type=int, default=1, help='Number of MambaBlocks of identical configuration stacked in sequence for ssm-mamba-dropInMLP.')
    mambaTower_parser.add_argument('--global_pool', action='store_true', help='Apply global pooling to the output of the ssm_mamba_dropInMLP model - changes output shape from (B,L,D) to (B,D)')

    return mambaTower_parser

