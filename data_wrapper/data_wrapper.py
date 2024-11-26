from dynabench.dataset import DynabenchIterator
from torch.utils.data import DataLoader
from typing import Tuple
from argparse import Namespace


def get_datasets(
    args: Namespace,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    if args.dataset == "dynabench":

        rollout = args.lookback

        train_iterator = DynabenchIterator(
            split="train",
            equation=args.equation,
            structure=args.structure,
            resolution=args.resolution,
            base_path=args.data_dir,
            lookback=args.lookback,
            rollout= rollout,
        )
        val_iterator = DynabenchIterator(
            split="val",
            equation=args.equation,
            structure=args.structure,
            resolution=args.resolution,
            base_path=args.data_dir,
            lookback=args.lookback,
            rollout= rollout,
            download=args.download,
        )
        test_iterator = DynabenchIterator(
            split="test",
            equation=args.equation,
            structure=args.structure,
            resolution=args.resolution,
            base_path=args.data_dir,
            lookback=args.lookback,
            rollout=rollout,
            download=args.download,
        )

        train_loader = DataLoader(train_iterator, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_iterator, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_iterator, batch_size=args.batch_size, shuffle=False)
    else: raise ValueError("dataset invalid or not implemented")

    return train_loader, val_loader, test_loader
