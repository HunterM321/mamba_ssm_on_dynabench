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

def create_dynabench_dataset_parser():
    dataset_parser = argparse.ArgumentParser(description="Configuration options for the Dynabench dataset")

    # Equation to use
    dataset_parser.add_argument(
        "--equation", type=str, default="wave",
        choices=["advection", "burgers", "gasdynamics", "kuramotosivashinsky", "reactiondiffustion", "wave"],
        help="The equation to use. Choices are 'advection', 'burgers', 'gasdynamics', 'kuramotosivashinsky', 'reactiondiffustion', or 'wave'."
    )

    # Structure of the dataset
    dataset_parser.add_argument(
        "--structure", type=str, default="cloud", choices=["cloud", "grid"],
        help="The structure of the dataset. Choices are 'cloud' or 'grid'."
    )

    # Resolution of the dataset
    dataset_parser.add_argument(
        "--resolution", type=str, default="low",
        choices=["low", "medium", "high", "full"],
        help=("The resolution of the dataset. Choices are 'low', 'medium', 'high', or 'full'.\n"
              "Low resolution: 225 points (15x15 grid for 'grid' structure),\n"
              "Medium resolution: 484 points (22x22 grid),\n"
              "High resolution: 900 points (30x30 grid),\n"
              "Full resolution: 64x64 grid.")
    )

    # Lookback for input timesteps
    dataset_parser.add_argument(
        "--lookback", type=int, default=1,
        help="Number of timesteps to use for the input data. Defaults to 1."
    )

    # Rollout for target timesteps
    dataset_parser.add_argument(
        "--rollout", type=int, default=1,
        help="Number of timesteps to use for the target data. Defaults to 1."
    )

    return dataset_parser


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

