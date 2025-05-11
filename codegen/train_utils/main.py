from .triplet_dataset import get_triplet_dataloaders
from .log_utils import load_logger
from .train_utils import train as train_model_utils
from .models import load_model_and_tokenizer

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer

def train_model(input_path, output_path, pretrain_model, report_to=None, embedding_dim=10, batch_size=32, n_epochs=1, lr=0.0001, validation_split=0.2, accumulation_steps=1):
    """
    Trains the model from scratch, saves it to the specified path.

    Args:
        edges_path (str): Path to the edges table (CSV).
        nodes_path (str): Path to the nodes table (CSV).
        output_path (str): Path to save the trained model.
        embedding_dim (int): Dimension of the embeddings.
        batch_size (int): Batch size for training.
        n_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        validation_split (float): Fraction of data to use for validation.
    """
    logger = load_logger(report_to)

    model, tokenizer = load_model_and_tokenizer(pretrain_model)

    # Create DataLoaders
    train_loader, val_loader = get_triplet_dataloaders(
        input_path,
        batch_size,
        tokenizer=tokenizer,
        validation_split=validation_split
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print("Lr = ", lr)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    train_model_utils(model, optimizer, train_loader, val_loader, device, logger, n_epochs, accumulation_steps)

    # Save the Trained Model
    model.save_model(output_path)
    print(f"Trained model saved to {output_path}")