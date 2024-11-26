from dynabench.dataset import DynabenchIterator, download_equation
from models.cnn_ssm import MambaCNNMOL

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from tqdm import tqdm
import wandb

Apple_computer = True
if Apple_computer:
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

lookback = 3
rollout = 3
epoch = 10

wandb.init(
    project="CPEN355",

    config={
    "auther": "hunter",
    "architecture": "CNN+SSM",
    "dataset": "Dynabench",
    "epochs": epoch,
    "lookback": lookback,
    "rollout": rollout,
    }
)

advection_train_iterator = DynabenchIterator(split="train",
                                             equation='advection',
                                             structure='grid',
                                             resolution='low',
                                             lookback=lookback,
                                             rollout=rollout)

train_loader = DataLoader(advection_train_iterator, batch_size=32, shuffle=True)

model = MambaCNNMOL(input_size=lookback,
                 output_size=rollout,
                 hidden_layers=3,
                 hidden_channels=9).to(device)
model.eval()

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(epoch):
    model.train()

    losses = []
    # Use tqdm for the outer loop to show epoch progress
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{10}", unit="batch") as pbar:
        for i, (x, y, p) in enumerate(train_loader):
            x, y = x[:, :, 0].float().to(device), y[:, :, 0].float().to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            
            loss = criterion(y_pred, y)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

            # Update the progress bar with loss information
            pbar.set_postfix({"Loss": loss.item()})
            pbar.update(1)
    
    mean_loss = np.mean(losses)
    wandb.log({'train_loss': mean_loss})

advection_test_iterator = DynabenchIterator(split="test",
                                            equation='advection',
                                            structure='grid',
                                            resolution='low',
                                            lookback=3,
                                            rollout=3)

test_loader = DataLoader(advection_test_iterator, batch_size=32, shuffle=False)

loss_values = []
with tqdm(total=len(test_loader), desc="Testing", unit="batch") as pbar:
    for i, (x, y, p) in enumerate(test_loader):
        x, y = x[:, :, 0].float().to(device), y[:, :, 0].float().to(device)
        y_pred = model(x, t_eval=range(17))
        loss = criterion(y_pred, y)
        
        # Append the loss to the list
        loss_values.append(loss.item())

        wandb.log({'test_loss': loss.item()})
        
        # Update the progress bar with loss information
        pbar.set_postfix({"Loss": loss.item()})
        pbar.update(1)

print(f"Mean Loss: {sum(loss_values) / len(loss_values)}")
