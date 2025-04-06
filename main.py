import argparse
import os

from dataset_utils.main import create_dataset

from train_utils.main import train_model

def _create_dataset(dataset_path):
    create_dataset(dataset_path)


def _train_model(output_path, dataset_path, pretrain_model, lr, n_epochs, report_to):
    train_model(
        output_path=output_path,
        input_path=dataset_path,
        pretrain_model=pretrain_model,
        lr=lr,
        n_epochs=n_epochs,
        report_to=report_to
    )


def main():
    parser = argparse.ArgumentParser(description="")

    # Create subparsers for different modes
    subparsers = parser.add_subparsers(title="modes", dest="mode", help="Operating mode")

    # Dataset mode parser
    dataset_parser = subparsers.add_parser("dataset", help="Create dataset")
    dataset_parser.add_argument("dataset_path", help="Path to store the created dataset")

    # Train mode parser
    train_parser = subparsers.add_parser("train", help="Download and train model")
    train_parser.add_argument("-o", "--output_path", help="Path to store the trained model", required=True)
    train_parser.add_argument("--dataset_path", help="Path to the dataset to train on", required=True)
    train_parser.add_argument("--pretrain_model", help="Name of hugging face model to load", required=True)

    train_parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="learning rate")
    train_parser.add_argument("-n", "--n_epochs", type=int, default=5, help="number of training epochs")

    train_parser.add_argument("--report_to", type=str, default="none", choices=["wandb", "none", "print"], help="Reporting backend.")


    args = parser.parse_args()

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
        )
    else:
        parser.print_help()  # If no mode is specified, print help

if __name__ == "__main__":
    main()
