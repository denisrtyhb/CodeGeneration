from TrainModel import dataset, train_utils, logging
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return model, tokenizer

def train_model(input_path, output_path, pretrain_model, report_to=None, embedding_dim=10, batch_size=32, n_epochs=1, lr=0.001, validation_split=0.2):
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
    logger = logging.load_logger(report_to)

    model, tokenizer = load_model_and_tokenizer(pretrain_model)

    # Create DataLoaders
    train_loader, val_loader = dataset.get_dataloaders(
        f"{input_path}/edges.csv",
        f"{input_path}/nodes.csv",
        batch_size,
        tokenizer=tokenizer,
        validation_split=validation_split
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_utils.train_model(model, optimizer, train_loader, val_loader, device, logger, n_epochs)

    # Save the Trained Model
    torch.save(model.state_dict(), output_path)
    print(f"Trained model saved to {output_path}")