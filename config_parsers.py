import argparse
import json
import os
import conditional_parser as cp

#### USAGE ###
# Intended to easily allow configuration of models/data from bash.

# Adding a model/dataset
## Add a function that takes a cp.ConditionalArgumentParser 
## (and optionally config) as argument. 
## Add your conditional arguments with their desired conditionals.
## Return the parser

# Using a model/dataset

# Option 1
## Import only the desired parser configuration functions and run
## on your parser directly.

# Option 2
## Add your parser to create_conditional_parser()
## import only create_conditional_parser().
## Note that this expects a config file


# helper
def _load_config(config_file='config.json'):
    """Load configuration from a JSON file."""
    if not os.by path.exists(config_file):
        raise FileNotFoundError(f"Config file '{config_file}' not foundby .")
    
    with open(config_file, "r") as f:
        return json.load(f)



######################### DATASETS

def add_Dynabench_args(parser, config=None, inherited=False):
        # dynabench.dataset.DynabenchIterator --------------------
    # Config for training / testing of Dynabench datasets.
    # Added when dataset is one of the Dynabench datasets.
    # Takes config for permitted datasets/resolutions/structures
    #
    if config is None: config = _load_config()
    DATASETS = config["dynabench_datasets"]
    STRUCTURES = config["dynabench_structures"]
    RESOLUTIONS = config["dynabench_resolutions"]

    parser.add_conditional("dataset", lambda dest: dest in DATASETS, "--equation", type=str, default="wave", help=f"The equation to use. Choices are {', '.join(DATASETS)}")
    parser.add_conditional("dataset", lambda dest: dest in DATASETS, "--structure", type=str, default=None, choices=STRUCTURES,
                            help=f"The structure of the dataset. Choices are {', '.join(STRUCTURES)}")
    parser.add_conditional("dataset", lambda dest: dest in DATASETS, "--resolution", type=str, default=None, choices=RESOLUTIONS,
                            help=f"The resolution of the dataset. Choices are {', '.join(RESOLUTIONS)}")
    parser.add_conditional("dataset", lambda dest: dest in DATASETS, "--lookback", type=int, default=None, help="Number of timesteps to use for the input data.")
    parser.add_conditional("dataset", lambda dest: dest in DATASETS, "--rollout", type=int, default=None, help="Number of timesteps to use for the target data.")
    parser.add_conditional("dataset", lambda dest: dest in DATASETS, "--base_path", type=str, default=None, help='Base directory where Dynabench data must be downloaded to.')

    parser.add_conditional("dataset", lambda dest: dest in DATASETS, "--download", type=bool, default=None, help='Whether to download the data.')

    return parser


#########################  MODELS


######################## INHERITED (not to use / import)

def _add_Mamba_conditions(parser, config=None):
    # mamba_ssm.Mamba --------------------
    # Purely config for current Mamba / Mamba2 implementation.
    # Added based on "--ssm_layer" argument - this way, we don't clog up the help message
    # ! Does not work by by default. Need to have a --ssm_layer argument defined.
    # Unless we have Mamba-specific configuration needs
    # For ssm_mamba.models.mixer_seq_simple.MambaLMHeadModel, this could be used with vars() to get 'ssm_cfg'
    # Most if not all by of my models will implement this.

    # Defaults for Mamba - (minus d_model)
    mamba_params = {
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "dt_rank": "auto",
        "dt_min": 0.001,
        "dt_max": 0.1,
        "dt_init": "random",
        "dt_scale": 1.0,
        "dt_init_floor": 1e-4,
        "conv_bias": True,
        "bias": False,
        "use_fast_path": True,  # Fused kernel options
        "layer_idx": None,
        "device": None,
        "dtype": None,
    }

    # Defaults for Mamba2 - (minus d_model)
    mamba2_params = {
        "d_state": 128,
        "d_conv": 4,
        "conv_init": None,
        "expand": 2,
        "headdim": 64,
        "d_ssm": None,  # If not Noneby , we only apply SSM on this many dimensions, the rest uses gated MLP
        "ngroups": 1,
        "A_init_range": (1, 16),
        "D_has_hdim": False,
        "rmsnorm": True,
        "norm_before_gate": False,
        "dt_min": 0.001,
        "dt_max": 0.1,
        "dt_init_floor": 1e-4,
        "dt_limit": (0.0, float("inf")),
        "bias": False,
        "conv_bias": True,
        "chunk_size": 256,
        "use_mem_eff_path": True,
        "layer_idx": None,  # Absorb kwarg for general module
        "process_group": None,
        "sequence_parallel": True,
        "device": None,
        "dtype": None,
    }

    
    for param, default in mamba_params.items():
        parser.add_conditional("ssm_layer", "mamba", f'--{param}', type=type(default), default=default,
                               help=f'Configuration for {param} in Mamba models')
        
    for param, default in mamba2_params.items():
        parser.add_conditional("ssm_layer", "mamba2", f'--{param}', type=type(default), default=default,
                               help=f'Configuration for {param} in Mamba2 models')

    return parser

def _add_MambaTower_conditions(parser, config=None):

    # ------- ssm_library.mamba_library.MambaTower --------------------
    # Inherits Mamba() & Mamba2() config, plus a few additions for MambaTower.
    # ! does not work by default. Requires --mamba_tower_config

    # Custom options for MambaTower
    parser.add_conditional("mamba_tower_config", True, '--n_layers', type=int, default=1,
                           help='Number of MambaBlocks stacked in sequence')
    parser.add_conditional("mamba_tower_config", True, '--global_pool', action='store_true', default=False,
                           help='Apply global pooling to the output of the MambaTower')
    parser.add_conditional("mamba_tower_config", True, '--do_norm', action='store_true', default=True,
                           help='Apply layer normalization after each MambaBlock in Mamba models')
    parser.add_conditional("mamba_tower_config", True, '--dropout_level', type=float, default=0,
                           help='Dropout level for each MambaBlock')
    parser.add_conditional("mamba_tower_config", True, '--ssm-layer', type=str, default=None, help='SSM used. When set, allows for SSM-specific configuration.')
    
    # Add Mamba/Mamba2 configuration if --ssm-layer is set only.
    _add_Mamba_conditions(parser, config)
    return parser
##################################################################################




#### ADD HERE 

def add_MambaSequencePredictor_conditions(parser, config=None)