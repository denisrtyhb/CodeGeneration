import argparse
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from codegen.dataset_utils.main import create_multiple_datasets
from codegen.train_utils.main import train_model

def _create_dataset(dataset_path):
    create_multiple_datasets(dataset_path)


def _train_model(output_path, dataset_path, pretrain_model, lr, n_epochs, report_to, batch_size, accumulation_steps):
    train_model(
        output_path=output_path,
        input_path=dataset_path,
        pretrain_model=pretrain_model,
        lr=lr,
        n_epochs=n_epochs,
        report_to=report_to,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps
    )


def main():
    parser = argparse.ArgumentParser(description="")

    # Add distributed training arguments
    parser.add_argument("--local_rank", type=int, default=-1,
                      help="Local rank for distributed training")

    # Create subparsers for different modes
    subparsers = parser.add_subparsers(title="modes", dest="mode", help="Operating mode")

    # Dataset mode parser
    dataset_parser = subparsers.add_parser("dataset", help="Create dataset")
    dataset_parser.add_argument("-d", "--dataset_path", help="Path to store the created dataset")

    # Train mode parser
    train_parser = subparsers.add_parser("train", help="Download and train model")
    train_parser.add_argument("-o", "--output_path", help="Path to store the trained model", required=True)
    train_parser.add_argument("--dataset_path", help="Path to the dataset to train on", required=True)
    train_parser.add_argument("--pretrain_model", help="Name of hugging face model to load", required=True)

    train_parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="learning rate")
    train_parser.add_argument("-bs", "--batch_size", type=int, default=16, help="Batch size")
    train_parser.add_argument("-as", "--accumulation_steps", type=int, default=1, help="Accumulation steps")
    train_parser.add_argument("-n", "--n_epochs", type=int, default=5, help="number of training epochs")

    train_parser.add_argument("--report_to", type=str, default="none", choices=["wandb", "none", "print"], help="Reporting backend.")

    args = parser.parse_args()

    # Initialize distributed training

    if args.mode == "dataset":
        _create_dataset(args.dataset_path)
    elif args.mode == "train":
        _train_model(
            output_path=args.output_path,
            dataset_path=args.dataset_path,
            pretrain_model=args.pretrain_model,
            lr=args.learning_rate,
            n_epochs=args.n_epochs,
            report_to=args.report_to,
            batch_size=args.batch_size,
            accumulation_steps=args.accumulation_steps
        )
    else:
        parser.print_help()  # If no mode is specified, print help

if __name__ == "__main__":
    main()
