import argparse

def create_main_parser():
    # Main parser with global arguments
    main_parser = argparse.ArgumentParser(description="Configure model training")
    main_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    main_parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")

    # Create a subparser for each model
    subparsers = main_parser.add_subparsers(dest="model", required=True, help="Choose the model to configure")

    # Add model-specific parsers as subparsers
    add_transformer_parser(subparsers)
    add_cnn_parser(subparsers)

    return main_parser

def add_transformer_parser(subparsers):
    # Transformer-specific parser
    transformer_parser = subparsers.add_parser("transformer", help="Transformer model configuration")
    transformer_parser.add_argument("--heads", type=int, required=True, help="Number of attention heads")
    transformer_parser.add_argument("--hidden_dim", type=int, required=True, help="Hidden dimension size")

def add_cnn_parser(subparsers):
    # CNN-specific parser
    cnn_parser = subparsers.add_parser("cnn", help="CNN model configuration")
    cnn_parser.add_argument("--stride", type=int, required=True, help="Stride size for convolution")

# Main function to parse arguments
if __name__ == "__main__":
    parser = create_main_parser()
    args = parser.parse_args()

    # Access parsed arguments
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    
    # Model-specific arguments
    if args.model == "transformer":
        print(f"Heads: {args.heads}")
        print(f"Hidden Dimension: {args.hidden_dim}")
    elif args.model == "cnn":
        print(f"Stride: {args.stride}")
