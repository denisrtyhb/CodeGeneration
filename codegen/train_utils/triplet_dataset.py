import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
import os
import random
from tqdm import tqdm

class GraphTripletDataset(Dataset):
    def __init__(self, edges_path, nodes_path):
        self.edges_df = pd.read_csv(edges_path)
        edges_count_before = len(self.edges_df)
        unk_count = sum(self.edges_df['type'] == 'unknown')

        self.nodes_df = pd.read_csv(nodes_path)
        node_count = len(self.nodes_df)
        self.nodes_list = self.nodes_df['node'].values
        self.node_to_source = dict(zip(self.nodes_df['node'], self.nodes_df['source_code']))

        print(f"Loaded graph dataset. Number of nodes: {node_count}.")

    def get_functions(self):
        return self.nodes_df.iterrows()

    def __len__(self):

        return len(self.edges_df)

    def get_node_to_source(self, node):
        res = self.node_to_source.get(node, "")
        if res is None or res == "" or (type(res) == float):
            res = "nothing"
        return res

    def __getitem__(self, idx):
        node1 = self.edges_df['node1'][idx]
        node2 = self.edges_df['node2'][idx]

        source_code1 = self.get_node_to_source(node1)  # Handle missing nodes gracefully
        source_code2 = self.get_node_to_source(node2)  # Handle missing nodes gracefully

        # random node from self.nodes_list
        node3 = random.choice(self.nodes_list)
        source_code3 = self.get_node_to_source(node3)  # Handle missing nodes gracefully

        return node1, node2, node3, source_code1, source_code2, source_code3

class MergeDataset(Dataset):
    """
    A dataset that merges multiple torch datasets into a single dataset.

    Args:
        datasets (list): A list of torch datasets to merge.  All datasets
                         are expected to have a `__len__` method.
    """

    def __init__(self, datasets):
        assert len(datasets) != 0, "Need at least one dataset for merging"
        self.datasets = datasets
        self.cumulative_sizes = self.cumsum(self.datasets)
        self.total_length = self.cumulative_sizes[-1]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)  # Use len() to get the length of the Dataset
            r.append(l + s)
            s += l
        return r

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_length:
            raise IndexError("Index out of range")


        dataset_idx = 0
        for i in range(len(self.datasets)):
            if idx < self.cumulative_sizes[i]:
                dataset_idx = i
                break

        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return self.datasets[dataset_idx][sample_idx]

def make_collate_fn(tokenizer):
    def collate_fn(batch):

        node1, node2, node3, source_code1, source_code2, source_code3 = zip(*batch)

        # TODO: wtf
        # tokenized_1 = tokenizer(list(strings1), padding=True, truncation=True, return_tensors="pt")
        # tokenized_2 = tokenizer(list(strings2), padding=True, truncation=True, return_tensors="pt")
        # try:
        tokenized_list = []
        for source_code in [source_code1, source_code2, source_code3]:
            tokenized = tokenizer(list(source_code), padding=True, truncation=True, max_length=256, return_tensors="pt")
            tokenized_list.append(tokenized)
        
        tokenized_1, tokenized_2, tokenized_3 = tokenized_list

        return (
            tokenized_1['input_ids'],
            tokenized_1['attention_mask'],

            tokenized_2['input_ids'],
            tokenized_2['attention_mask'],

            tokenized_3['input_ids'],
            tokenized_3['attention_mask']
        )
    return collate_fn

def get_dataset(base_path):
    edges_path = os.path.join(base_path, "edges.csv")
    nodes_path = os.path.join(base_path, "nodes.csv")

    if not os.path.isfile(edges_path) or not os.path.isfile(nodes_path):
        return None
    try:
        return GraphTripletDataset(edges_path, nodes_path)
    except Exception as e:
        print(f"Error loading dataset from {base_path}: {e}")
        return None

def train_test_split(dataset_list, split_size):
    random.shuffle(dataset_list)

    total_size = sum(map(lambda x: len(x), dataset_list))
    train_size = 0
    sep = 0

    while train_size < total_size * (1 - split_size):
        train_size += len(dataset_list[sep])
        sep += 1

    print(len(dataset_list))
    
    train = MergeDataset(dataset_list[:sep])
    test = MergeDataset(dataset_list[sep:])
    
    assert len(train) != 0 and len(test) != 0

    return train, test

def get_all_datasets(base_path):
    
    dataset_list = []
    for folder in tqdm(os.listdir(base_path), desc="Listing datasets for loading"):
        cur = get_dataset(os.path.join(base_path, folder))
        if cur is not None:
            dataset_list.append(cur)
    return dataset_list

def get_triplet_dataloaders(base_path, batch_size, tokenizer, validation_split=0.2, shuffle=True, random_seed=42):

    dataset_list = get_all_datasets(base_path)
    
    print("Number of datasets:", len(dataset_list))

    train_dataset, val_dataset = train_test_split(dataset_list, validation_split)
    
    collate_fn = make_collate_fn(tokenizer)
    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False) # No need to shuffle validation

    return train_loader, val_loader