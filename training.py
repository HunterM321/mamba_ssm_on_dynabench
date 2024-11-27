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
    parser.add_conditional('model', "MambaPatchMOL", "--mamba_struct", default="seq", type=str)

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


def train_loop(args, model, train_loader, val_loader):

    apple_computer = False
    if apple_computer:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    criterion = torch.nn.MSELoss()
    optimizer = None

    for epoch in range(args.epochs):    
            
        train_bar = tqdm(train_loader, desc=f"Epoch #{epoch} Training", leave=False)
        model.train()

        epoch_train_loss = 0.0
        for x, y, p in train_bar:

            x, y, p = x.to(dtype=torch.float32, device=device), y.to(dtype=torch.float32, device=device), p.to(dtype=torch.float32, device=device)

            # If we predict only next time step we will take the first ground truth time step
            if args.training_setting == "nextstep": 
                y =y[:,[0],:]

            # The Mamba Patch model does not define linear layers and parameters until AFTER the first forward pass is ran
            # So we must have the two if statements below to ensure that this training loop is generalizable.
            if optimizer != None: 
                optimizer.zero_grad()

            # Predict Either next step or next ROLLOUT steps
            y_pred = model(x)

            # If this is MambaPatch and first datapoint it sees then it has no parameters
            if optimizer == None:
                optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)             

            # Calculate MSE, calculate gradients, update weights
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            # Update variables relating to WandB logging purposes
            epoch_train_loss += loss.item()

            train_bar.set_postfix_str(f"Loss: {loss.item()}")

        # Log epoch training loss to Wandb. Divide Epoch loss with number of batches
        epoch_train_loss = epoch_train_loss/len(train_bar)
        wandb.log({"Epoch Training Loss": epoch_train_loss})

        # Default is log_interval == 1 so we run validation every epoch.
        if epoch % args.log_interval == 0:

            val_bar = tqdm(val_loader, desc="Validation", leave=False)  # Add a tqdm progress bar
            model.eval() # Running inference on validation dataset
            
            epoch_val_loss = 0.0
            with torch.no_grad(): # Ensures we do not train on validation data
                for x_v, y_v, p_v in val_bar:
                    
                    optimizer.zero_grad()

                    x_v, y_v, p_v = x_v.to(dtype=torch.float32, device=device), y_v.to(dtype=torch.float32, device=device), p_v.to(dtype=torch.float32, device=device)
                    
                    if args.training_setting == "nextstep": 
                        y_v =y_v[:,[0],:] # remove all but next step, but keep time axis

                    y_pred_v = model(x_v)

                    loss_v = criterion(y_pred_v, y_v)
                    epoch_val_loss += loss_v.item()

                    val_bar.set_postfix_str(f"Current val loss: {loss_v.item()}")

                epoch_val_loss = epoch_val_loss/len(val_bar)
                wandb.log({"Epoch Validation Loss": epoch_val_loss})
                # print(f"Average val loss: {epoch_val_loss:.4f}")

def test(args, model, test_loader):

    apple_computer = False
    if apple_computer:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    test_bar = tqdm(test_loader, desc="Testing", leave=False) 

    # For a lookback/rollout of N, S-to-S predicts the next N time steps all at once with only
    # the original input of time length N.
    if args.training_setting == "seqtoseq": # seqtoseq testing

        seq_to_seq_test_loss = 0

        with torch.no_grad():
            for x_v, y_v, p_v in test_bar:

                x_v, y_v, p_v = x_v.to(dtype=torch.float32, device=device), y_v.to(dtype=torch.float32, device=device), p_v.to(dtype=torch.float32, device=device)

                optimizer.zero_grad()
                y_pred_v = model(x_v).detach()
                loss_v = criterion(y_pred_v, y_v)
                seq_to_seq_test_loss += loss_v

                test_bar.set_postfix_str(f"Current test loss: {loss_v.item()}")

            seq_to_seq_test_loss /= len(test_bar)
            wandb.log({"Seq-to-Seq Test Loss": test})
            print(f"Average test loss: {seq_to_seq_test_loss:.4f}")


    # For lookback/rollout of N, nextstep also predicts the next N time steps, but does this by predicting only the immediate
    # next time step, and using the predicted time steps as part of the input in predicting future time steps. I.e. it autoregressively
    # predicts the next N timesteps.
    elif args.training_setting == "nextstep": # use rollout 

        # A 1D vector where the value at index i is the loss from predicting y_pred at i
        losses_over_rollout = torch.zeros(args.lookback).to(device)

        with torch.no_grad():
            for x_v, y_v, p_v in test_bar:

                x_v, y_v, p_v = x_v.to(dtype=torch.float32, device=device), y_v.to(dtype=torch.float32, device=device), p_v.to(dtype=torch.float32, device=device)

                optimizer.zero_grad()

                # The code below allows us to regressively use the next predicted time step as an input for further predictions

                # Meaning for a lookback (and rollout) of N, column i in y_pred is the predicted immediate next time step 
                # given i previously predicted outputs, and N-i original inputs (which preceeds the i previously predicted outputs).
                y_preds_v = torch.cat(
                    [(x := torch.cat((x_v[:, 1:], (pred := model(x_v))), dim=1))[:, -1:] for _ in range(args.lookback)],
                    dim=1
                )


                losses_v = torch.stack(
                    [criterion(y_preds_v[:, i], y_v[:, i]) for i in range(args.lookback)],
                    dim=0
                )

                losses_over_rollout += losses_v
                test_bar.set_postfix_str(f"Current test loss at each step: {losses_v}")

            losses_over_rollout /= len(test_bar)

            # Create a bar plot with indices and loss
            data = [[i, value.item()] for i, value in enumerate(losses_over_rollout)]
            table = wandb.Table(data=data, columns=["Index", "Loss"])

            wandb.log({"Regressive S2S Test Loss": wandb.plot.bar(
                table, "Index", "Loss", title="Loss Accumulation Over Consecutive Predictions"
            )})

            # print(f"Average test loss: {losses_over_rollout}")

def main(args):

    wandb.init(
            project="CPEN355",
            config=create_config(args),
    )

    # data and model
    train_loader, val_loader, test_loader = get_datasets(args)
    model = get_model(args).to('cuda:0')

    # training prep
    # optimizer, criterion = get_training_tools(args, model)

    if args.mode in ["train", "both"]:
        train_loop(args, model, train_loader, val_loader)
    if args.mode in ["test", "both"]:
        test(args, model, test_loader)



if __name__ == "__main__":
    # Parse arguments
    parser = create_arg_parser()
    args = parser.parse_args()

    main(args)

