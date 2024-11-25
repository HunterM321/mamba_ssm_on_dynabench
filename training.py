import sys
import os

# Add the parent directory to the Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "data_wrapper")))

from data_wrapper.data_wrapper import get_datasets
from utils.model_wrapper import get_model

import argparse
import conditional_parser as cp
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import models.mamba_library as ml
from config_parsers import create_dynabench_dataset_parser, create_MambaTower_parser

from dynabench.dataset import DynabenchIterator

with open("config.json", "r") as f:
    config = json.load(f)
EQUATIONS = config["dynabench"]["dynabench_equations"]
STRUCTURES = config["dynabench"]["dynabench_structures"]
RESOLUTIONS = config["dynabench"]["dynabench_resolutions"]

def create_arg_parser():
    parser = cp.ConditionalArgumentParser(description="Sequence model testing on a dynamical system dataset")

    # ------- General Configuration -------
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True,
                        help="Mode of operation: 'train' or 'test'. In 'train' mode, training and validation data are used. In 'test' mode, only test data is used.")

    # ------- Dataset Configuration -------
    parser.add_argument('--dataset', type=str, )
    parser.add_argument("--equation", type=str, choices=EQUATIONS, default="wave",
                        help="The equation to use in the Dynabench dataset.")
    parser.add_argument("--structure", type=str, choices=STRUCTURES, default="cloud",
                        help="The structure of the dataset (cloud or grid).")
    parser.add_argument("--resolution", type=str, choices=RESOLUTIONS, default="low",
                        help="The resolution of the dataset.")
    parser.add_argument("--lookback", type=int, default=1, help="Number of timesteps for input data.")
    parser.add_argument("--rollout", type=int, default=1, help="Number of timesteps for target data.")
    parser.add_argument("--download", action="store_true", help="If set, downloads the data.")

    # ------- I/O Configuration -------
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory where the data is stored.")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints",
                        help="Path to save model checkpoints.")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save output logs or predictions.")

    # ------- Training Configuration -------
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--optimizer", type=str, choices=["adam", "sgd"], default="adam",
                        help="Optimizer to use for training.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay (L2 regularization).")
    parser.add_argument("--loss", type=str, choices=["mse", "cross_entropy"], default="mse",
                        help="Loss function to use for training.")
    parser.add_argument("--log_interval", type=int, default=10, help="Interval (in batches) to log training progress.")

    # ------- Model Configuration -------
    # subparsers = parser.add_subparsers(dest="model", required=True, help="Choose the model to configure")

    # Transformer-specific arguments
    # transformer_parser = subparsers.add_parser("transformer", help="Transformer model configuration")
    # transformer_parser.add_argument("--heads", type=int, required=True, help="Number of attention heads")
    # transformer_parser.add_argument("--hidden_dim", type=int, required=True, help="Hidden dimension size")
    # transformer_parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    # transformer_parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # CNN-specific arguments
    # cnn_parser = subparsers.add_parser("cnn", help="CNN model configuration")
    # cnn_parser.add_argument("--num_filters", type=int, required=True, help="Number of convolutional filters")
    # cnn_parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size for convolutions")
    # cnn_parser.add_argument("--stride", type=int, default=1, help="Stride size for convolution")
    # cnn_parser.add_argument("--padding", type=int, default=1, help="Padding for convolution layers")
    # cnn_parser.add_argument("--pooling", type=str, choices=["max", "avg"], default="max", help="Pooling type")

    return parser

def main():

    train_loader, val_loader, test_loader = get_datasets(args)
    model = model_wrapper.get_model(args)

if __name__ == "__main__":
    # Parse arguments
    parser = create_arg_parser()
    args = parser.parse_args()

    main(args)