from codegen.utils import dataset


if __name__ == "__main__":
    dataset = dataset.GraphDataset(
        edges_path="../data/example2/edges.csv",
        nodes_path="../data/example2/nodes.csv",
    )

    cnt = 0
    for node1, node2, source_code1, source_code2 in dataset:
        print(node1, source_code1, sep='\n', end='\n\n\n')
        cnt += 1
        if cnt == 5:
            break