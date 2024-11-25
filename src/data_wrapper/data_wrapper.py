from dynabench.dataset import DynabenchIterator
from torch.utils.data import DataLoader
from typing import Tuple
from conditional_parser import ConditionalArgumentParser


def get_datasets(
    args: ConditionalArgumentParser,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_iterator = DynabenchIterator(
        split="train",
        equation=args.equation,
        structure=args.structure,
        resolution=args.resolution,
        base_path=args.data_dir,
        lookback=args.lookback,
        rollout=args.rollout,
    )
    val_iterator = DynabenchIterator(
        split="val",
        equation=args.equation,
        structure=args.structure,
        resolution=args.resolution,
        base_path=args.data_dir,
        lookback=args.lookback,
        rollout=args.rollout,
        download=args.download,
    )
    test_iterator = DynabenchIterator(
        split="test",
        equation=args.equation,
        structure=args.structure,
        resolution=args.resolution,
        base_path=args.data_dir,
        lookback=args.lookback,
        rollout=args.rollout,
        download=args.download,
    )

    train_loader = DataLoader(train_iterator, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_iterator, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_iterator, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
