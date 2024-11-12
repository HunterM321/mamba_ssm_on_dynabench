import argparse
import json
from dynabench.dataset import download_equation

with open("config.json", "r") as f:
    config = json.load(f)
DATASETS = config["dynabench_datasets"]
STRUCTURES = config["dynabench_structures"]
RESOLUTIONS = config["dynabench_resolutions"]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Download datasets with specific configurations")

    # Argument for selecting multiple datasets
    parser.add_argument(
        '--datasets', nargs='+', choices=DATASETS + ['ALL'], required=True,
        help=f"List of datasets to download. Choices are: 'ALL' or {', '.join(DATASETS)}",
    )

    # Arguments for structure and resolution
    parser.add_argument(
        '--structure', choices=STRUCTURES, default='grid',
        help="Data structure to use for each dataset (default: grid). Choices are: 'grid' or 'cloud'"
    )
    parser.add_argument(
        '--resolution', choices=RESOLUTIONS, default='low',
        help="Data resolution to use for each dataset (default: low). Choices are: 'low', 'medium', or 'high'"
    )

    # Directory where datasets should be saved
    parser.add_argument(
        '--data_dir', type=str, default='data',
        help="Directory where datasets will be saved (default: '/data')"
    )

    return parser.parse_args()

def main():
    args = parse_arguments()

    datasets = args.datasets if 'ALL' not in args.datasets else DATASETS

    # Download each selected dataset with the specified configuration
    for dataset in datasets:
        print(f'Downloading {dataset}-{args.structure}-{args.resolution} to {args.data_dir}...')
        download_equation(
            equation=dataset,
            structure=args.structure,
            resolution=args.resolution,
            data_dir=args.data_dir
        )

if __name__ == "__main__":
    main()