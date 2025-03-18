from TrainModel import dataset, train_utils
import torch
import torch.nn as nn
import torch.optim as optim

def train_model(input_path, model_path, embedding_dim=10, batch_size=32, num_epochs=1, learning_rate=0.001, validation_split=0.2):
    """
    Trains the model from scratch, saves it to the specified path.

    Args:
        edges_path (str): Path to the edges table (CSV).
        nodes_path (str): Path to the nodes table (CSV).
        model_path (str): Path to save the trained model.
        embedding_dim (int): Dimension of the embeddings.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        validation_split (float): Fraction of data to use for validation.
    """

    # Create DataLoaders
    train_loader, val_loader = dataset.get_dataloaders(
        f"{input_path}/edges.csv",
        f"{input_path}/nodes.csv",
        batch_size, validation_split=validation_split)

    # Instantiate Model, Optimizer, and Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class DummyModel(nn.Module):
        def __init__(self, embedding_dim=10):
            super().__init__()
            self.embedding_dim = embedding_dim
            self.linear = nn.Linear(100, embedding_dim) #Dummy operation simulating embedding

        def forward(self, texts):
            # Dummy operation. Replace with your actual model logic.
            #  Convert texts into embeddings
            #  For simplicity, let's just return random tensors of the required size
            batch_size = len(texts)

            #Simulate processing and create a dummy vector per source code
            dummy_vectors = torch.randn(batch_size, 100)

            embeddings = self.linear(dummy_vectors)
            return embeddings

    model = DummyModel(embedding_dim=embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the Model
    train_utils.train_model(model, optimizer, train_loader, val_loader, device, num_epochs)

    # Save the Trained Model
    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved to {model_path}")