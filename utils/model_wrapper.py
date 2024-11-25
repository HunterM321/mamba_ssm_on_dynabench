import sys
import os

# Add the parent directory (C:\School_Code\CPEN355) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the models module
import models

def get_model(args):

    if args.model_type == "patching":
        model = MambaPatchMOL(args.patch_time_handling)
    elif args.model_type == "cnn-ssm":
        model = MambaCNNMOL(args.cnn_smm_time_handling)
    else:
        raise ValueError(f"Model type {model_type} not recognized.")

    return model
