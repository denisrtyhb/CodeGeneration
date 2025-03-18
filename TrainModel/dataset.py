import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

class GraphDataset(Dataset):
    def __init__(self, edges_path, nodes_path):
        self.edges_df = pd.read_csv(edges_path)
        self.nodes_df = pd.read_csv(nodes_path)
        self.node_to_source = dict(zip(self.nodes_df['node'], self.nodes_df['source_code']))


    def __len__(self):

        return len(self.edges_df)

    def __getitem__(self, idx):
        node1 = self.edges_df['node1'][idx]
        node2 = self.edges_df['node2'][idx]

        source_code1 = self.node_to_source.get(node1, "")  # Handle missing nodes gracefully
        source_code2 = self.node_to_source.get(node2, "")  # Handle missing nodes gracefully

        return node1, node2, source_code1, source_code2

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def collate_fn(batch):
    global tokenizer

    strings1, strings2, strings3, strings4 = zip(*batch)

    tokenized_1 = tokenizer(list(strings1), padding=True, truncation=True, return_tensors="pt")
    tokenized_2 = tokenizer(list(strings2), padding=True, truncation=True, return_tensors="pt")
    tokenized_3 = tokenizer(list(strings3), padding=True, truncation=True, return_tensors="pt")
    tokenized_4 = tokenizer(list(strings4), padding=True, truncation=True, return_tensors="pt")

    return (
        tokenized_1['input_ids'],
        tokenized_2['input_ids'],
        tokenized_3['input_ids'],
        tokenized_4['input_ids']
    )


def get_dataloaders(edges_path, nodes_path, batch_size, validation_split=0.2, shuffle=True, random_seed=42):

    full_dataset = GraphDataset(edges_path, nodes_path)

    # Calculate the sizes of the training and validation sets
    val_size = int(len(full_dataset) * validation_split)
    train_size = len(full_dataset) - val_size

    # Split the dataset into training and validation sets
    generator = torch.Generator().manual_seed(random_seed) # Use random seed for reproducibility
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Create the DataLoaders
    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False) # No need to shuffle validation

    return train_loader, val_loader