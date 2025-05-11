
def get_dataset(base_path):
    edges_path = os.path.join(base_path, "edges.csv")
    nodes_path = os.path.join(base_path, "nodes.csv")

    if not os.path.isfile(edges_path) or not os.path.isfile(nodes_path):
        return None
    return GraphDataset(edges_path, nodes_path)