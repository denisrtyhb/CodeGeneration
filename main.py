import argparse
import os

from DatasetCreation.main import create_dataset

from TrainModel.main import train_model

def _create_dataset(dataset_path):
    create_dataset(dataset_path)


def _train_model(model_path, dataset_path):
    train_model(dataset_path, model_path)


def main():
    parser = argparse.ArgumentParser(description="")

    # Create subparsers for different modes
    subparsers = parser.add_subparsers(title="modes", dest="mode", help="Operating mode")

    # Dataset mode parser
    dataset_parser = subparsers.add_parser("dataset", help="Create dataset")
    dataset_parser.add_argument("dataset_path", help="Path to store the created dataset")

    # Train mode parser
    train_parser = subparsers.add_parser("train", help="Download and train model")
    train_parser.add_argument("model_path", help="Path to store the downloaded model")
    train_parser.add_argument("dataset_path", help="Path to the dataset to train on")

    args = parser.parse_args()

    if args.mode == "dataset":
        _create_dataset(args.dataset_path)
    elif args.mode == "train":
        _train_model(args.model_path, args.dataset_path)
    else:
        parser.print_help()  # If no mode is specified, print help

if __name__ == "__main__":
    main()
