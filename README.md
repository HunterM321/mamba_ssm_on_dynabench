## Group 2: Evaluation of State-Space Models as efficient dynamical system solvers

To run the test and evaluation code you need to first download the dataset. You can do so by creating a python script in the level as this repo and paste the following command:
``` python
from dynabench.dataset import download_equation

download_equation('advection', structure='grid', resolution='low')
```
This will save the dataset in a directory called data in the same level as this repo. Then run any of the following command:
1. If you want to run sequence-to-sequence prediction:
   1. If you want to choose MambaCNNMOL as your dynamical systeml solver, run  ```python training.py --mode both --dataset dynabench --training_setting seqtoseq --lookback 5  --model MambaCNNMOL --input_size 5 --output_size 5 --hidden_layers 5 --hidden_channels 10 --epochs 10```.
   2. If you want to choose MambaPatchMOL as your dynamical systeml solver, run  ```python training.py --mode both --dataset dynabench --training_setting seqtoseq --lookback 5  --model MambaPatchMOL --patch_size 5 --n_layers 3 --time_handling keep --mamba_struct seq --epochs 10```. This runs the Mamba blocks in sequential fashion. To run the Mamba blocks in parallel fashion, change the ```--mamba_struct``` argument to ```parallel```.
2. If you want to run sequence-to-one prediction:
   1. If you want to choose MambaCNNMOL as your dynamical systeml solver, run  ```python training.py --mode both --dataset dynabench --training_setting nextstep --lookback 5  --model MambaCNNMOL --input_size 5 --output_size 5 --hidden_layers 5 --hidden_channels 10 --epochs 10```.
   2. If you want to choose MambaPatchMOL as your dynamical systeml solver, run  ```python training.py --mode both --dataset dynabench --training_setting nextstep --lookback 5 --epochs 10 --model MambaPatchMOL --patch_size 5 --n_layers 3 --time_handling poolmean --mamba_struct seq --epochs 10```. This runs the Mamba blocks in sequential fashion. To run the Mamba blocks in parallel fashion, change the ```--mamba_struct``` argument to ```parallel```.