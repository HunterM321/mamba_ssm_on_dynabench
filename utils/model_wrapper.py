import sys
import os

import models.MambaPatchMOL
import models.cnn_ssm

# Add the parent directory (C:\School_Code\CPEN355) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the models module
import models

def get_model(args):

    if args.model == "MambaPatchMOL":

        if (args.training_setting == "seqtoseq" and args.time_handling != "keep") or (args.training_setting == "nextstep" and args.time_handling not in ["last", "poolmax",  "poolmean"]): 
            raise ValueError(f"training object {args.training_setting} incompatible with desired architecture {args.time_handling}")
        
        model = models.MambaPatchMOL.MambaPatchMOL(args.patch_size, args.d_model, args.n_layers, args.time_handling)
    elif args.model == "MambaCNNMOL":

        if args.training_setting == "nextstep" and args.output_size != 1:
            raise ValueError(f"training object {args.training_setting} incompatible with desired rollout size {args.output_size}")
        
        model = models.cnn_ssm.MambaCNNMOL(args.input_size, args.output_size, args.hidden_layers, args.hidden_channels)
    else:
        raise ValueError(f"Model type {args.model} not recognized.")

    return model
