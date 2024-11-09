import os
import argparse

import torch
from torch import optim
from torch.utils.data import DataLoader


import mamba_library as ml
from dynabench.dataset import DynabenchIterator

# arg parse with arguments for 

#model
## hyper parameters

# training
## dataset
## training parameters

# I/O
## model checkpoint save paths
## data_dir

parser = argparse.ArgumentParser()

# ------- dataset config


# ------- I/O config


it = DynabenchIterator()
