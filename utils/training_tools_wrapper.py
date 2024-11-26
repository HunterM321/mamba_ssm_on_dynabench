import sys
import os
import torch.nn as nn
import torch.optim as optim

# Add the parent directory (C:\School_Code\CPEN355) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the models module

def get_training_tools(args, model):

    
    try:
        if args.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    except ValueError as e:
        print(f"Couldn't make optimizer. Will try again after first forward pass.")
        optimizer = None

    if args.loss == "mse":
        criterion = nn.MSELoss()
    else: raise ValueError('Invalid loss')
    
    return optimizer, criterion
