import sys
import os
import wandb

# Add the parent directory to the Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "data_wrapper")))

from data_wrapper.data_wrapper import get_datasets
from utils.model_wrapper import get_model
from utils.training_tools_wrapper import get_training_tools

import argparse
import conditional_parser as cp
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

with open("config.json", "r") as f:
    config = json.load(f)
EQUATIONS = config["dynabench"]["dynabench_equations"]
STRUCTURES = config["dynabench"]["dynabench_structures"]
RESOLUTIONS = config["dynabench"]["dynabench_resolutions"]

def create_arg_parser():
    parser = cp.ConditionalArgumentParser(description="Sequence model testing on a dynamical system dataset")

    # ------- General Configuration -------
    parser.add_argument("--mode", type=str, choices=["train", "test", "both"], required=True,
                        help="Mode of operation: 'train' or 'test'. In 'train' mode, training and validation data are used. In 'test' mode, only test data is used.")

    # ------- Dataset Configuration -------
    parser.add_argument('--dataset', type=str, default="dynabench")

    # Dynabench specific
    parser.add_conditional("dataset", "dynabench", "--equation", type=str, choices=EQUATIONS, default="advection",
                        help="The equation to use in the Dynabench dataset.")
    parser.add_conditional("dataset", "dynabench", "--structure", type=str, choices=STRUCTURES, default="grid",
                        help="The structure of the dataset (cloud or grid).")
    parser.add_conditional("dataset", "dynabench", "--resolution", type=str, choices=RESOLUTIONS, default="low",
                        help="The resolution of the dataset.")
    parser.add_conditional("dataset", "dynabench", "--training_setting", type=str, choices=["seqtoseq", "nextstep"])
    parser.add_conditional("dataset", "dynabench", "--lookback", type=int, default=1, help="Number of timesteps for input data.")
    # parser.add_conditional("dataset", "dynabench", "--rollout", type=int, default=1, help="Number of timesteps for target data.") # always use lookback
    parser.add_conditional("dataset", "dynabench", "--download", action="store_true", help="If set, downloads the data.")

    # ------- I/O Configuration -------
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory where the data is stored.")
    # parser.add_argument("--checkpoint_path", type=str, default="checkpoints",
    #                     help="Path to save model checkpoints.")
    # parser.add_argument("--output_dir", type=str, default="outputs",
    #                     help="Directory to save output logs or predictions.")

    # ------- Training Configuration -------
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--optimizer", type=str, choices=["adam", "sgd"], default="adam",
                        help="Optimizer to use for training.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay (L2 regularization).")
    parser.add_argument("--loss", type=str, choices=["mse", "cross_entropy"], default="mse",
                        help="Loss function to use for training.")
    parser.add_argument("--log_interval", type=int, default=1, help="Interval (in batches) to log training progress.")  

    # ------- Model Configuration -------
    parser.add_argument("--model", required=True, help="Choose the model to configure")

    # for MambaPatchMOL
    parser.add_conditional('model', "MambaPatchMOL", "--patch_size", default=5, type=int)
    parser.add_conditional('model', "MambaPatchMOL", "--d_model", default=16, type=int)
    parser.add_conditional('model', "MambaPatchMOL", "--n_layers", default=3, type=int)
    parser.add_conditional('model', "MambaPatchMOL", "--time_handling", required=True, type=str) # decided by objective

    # for MambaCNNMOL
    parser.add_conditional('model', "MambaCNNMOL", "--input_size", required=True, type=int)     # Lookback size
    parser.add_conditional('model', "MambaCNNMOL", "--output_size", required=True, type=int)    # Rollout size
    parser.add_conditional('model', "MambaCNNMOL", "--hidden_layers", default=3, type=int)
    parser.add_conditional('model', "MambaCNNMOL", "--hidden_channels", default=4, type=int)

    return parser

def create_config(args):
    """
    Creates a configuration dictionary for wandb based on parsed arguments.
    
    Args:
        args (Namespace): Parsed arguments from argparse.
    
    Returns:
        dict: A configuration dictionary for wandb.
    """
    return {
        # General Configuration
        "mode": args.mode,
        
        # Dataset Configuration
        "dataset": args.dataset,
        "dynabench": {
            "equation": args.equation if args.dataset == "dynabench" else None,
            "structure": args.structure if args.dataset == "dynabench" else None,
            "resolution": args.resolution if args.dataset == "dynabench" else None,
            "training_setting": args.training_setting if args.dataset == "dynabench" else None,
            "lookback": args.lookback if args.dataset == "dynabench" else None,
        },
        
        # Training Configuration
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "loss": args.loss,
        "log_interval": args.log_interval,

        # Model Configuration
        "model": args.model,
    }


def train_loop(args, model, optimizer, criterion, train_loader, val_loader):

    apple_computer = False
    if apple_computer:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for epoch in range(args.epochs):        
        train_bar = tqdm(train_loader, desc=f"Epoch #{epoch} Training", leave=False)
        model.train()
        for x, y, p in train_bar:

            x, y, p = x.to(dtype=torch.float32, device=device), y.to(dtype=torch.float32, device=device), p.to(dtype=torch.float32, device=device)

            if args.training_setting == "nextstep": 
                y =y[:,[0],:] # remove all but next step, but keep time axis
            
            if optimizer != None: optimizer.zero_grad()
            y_pred = model(x)
            if optimizer == None: optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) # required when model defined by forward()
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            train_bar.set_postfix_str(f"Loss: {loss.item()}")

        if epoch % args.log_interval == 0:
            val_loss = 0.0
            model.eval()

            val_bar = tqdm(val_loader, desc="Validation", leave=False)  # Add a tqdm progress bar

            with torch.no_grad():
                for x_v, y_v, p_v in val_bar:

                    x_v, y_v, p_v = x_v.to(dtype=torch.float32, device=device), y_v.to(dtype=torch.float32, device=device), p_v.to(dtype=torch.float32, device=device)

                    optimizer.zero_grad()
                    if args.training_setting == "nextstep": 
                        y_v =y_v[:,[0],:] # remove all but next step, but keep time axis
                    y_pred_v = model(x_v)
                    loss_v = criterion(y_pred_v, y_v)
                    val_loss += loss_v

                    val_bar.set_postfix_str(f"Current val loss: {loss_v.item()}")

                val_loss /= len(val_loader)
                print(f"Average val loss: {val_loss:.4f}")

def test(args, model, optimizer, criterion, test_loader):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    test_bar = tqdm(test_loader, desc="Testing", leave=False) 


    if args.training_setting == "seqtoseq": # seqtoseq testing

        test_loss = 0.0

        with torch.no_grad():
            for x_v, y_v, p_v in test_bar:

                x_v, y_v, p_v = x_v.to(dtype=torch.float32, device=device), y_v.to(dtype=torch.float32, device=device), p_v.to(dtype=torch.float32, device=device)

                optimizer.zero_grad()
                y_pred_v = model(x_v).detach()
                loss_v = criterion(y_pred_v, y_v)
                test_loss += loss_v

                test_bar.set_postfix_str(f"Current test loss: {loss_v.item()}")

                test_loss /= len(test_loader)
                print(f"Average test loss: {test_loss:.4f}")

    elif args.training_setting == "nextstep": # use rollout 

        losses_over_rollout = torch.zeros(args.lookback)

        with torch.no_grad():
            for x_v, y_v, p_v in test_bar:

                x_v, y_v, p_v = x_v.to(dtype=torch.float32, device=device), y_v.to(dtype=torch.float32, device=device), p_v.to(dtype=torch.float32, device=device)

                optimizer.zero_grad()

                y_preds_v = torch.cat(
                    [(x := torch.cat((x_v[:, 1:], (pred := model(x_v))), dim=1))[:, -1:] for _ in range(args.lookback)],
                    dim=1
                ).detach()

                losses_v = torch.stack(
                    [criterion(y_preds_v[:, i], y_v[:, i]) for i in range(args.lookback)],
                    dim=0
                )

                losses_over_rollout += losses_v

                test_bar.set_postfix_str(f"Current test loss at each step: {losses_v}")

            losses_over_rollout /= len(losses_over_rollout)
            print(f"Average test loss: {losses_over_rollout}")

def main(args):

    wandb.init(
            project="CPEN355",
            config=create_config(args),
        )

    # data
    train_loader, val_loader, test_loader = get_datasets(args)

    # model
    model = get_model(args)

    # training prep
    optimizer, criterion = get_training_tools(args, model)

    if args.mode in ["train", "both"]:

        train_loop(args, model, optimizer, criterion, train_loader, val_loader)

    if args.mode in ["test", "both"]:
        test(args, model, optimizer, criterion, test_loader)



if __name__ == "__main__":
    # Parse arguments
    parser = create_arg_parser()
    args = parser.parse_args()

    main(args)

