# import DatasetCreation.file_utils
from DatasetCreation import ast_utils, file_utils

import ast
from astmonkey import visitors
from tqdm import tqdm
import os
import fnmatch


def process_file(path, verbose=False):
    ast_tree = ast_utils.construct_ast_from_file(path)
    assert ast_tree, f"Failed to construct AST for {path}"

    visitor = ast_utils.GraphMakerVisitor()

    graph = visitor.construct_graph(ast_tree)

    if verbose:
        for name, edges, node in graph:
            source = visitors.to_source(node).replace('\\', '').strip()
            print("\n\n\n")
            print("Source code:")
            print(source)
            print("Edges:")
            print(name, "->", ', '.join(edges))
    return graph


def create_dataset(input_path, output_path=None, verbose=False):
    if output_path is None:
        output_path = input_path


    edges_path = f"{output_path}/edges.csv"
    node_path = f"{output_path}/nodes.csv"

    graph = []

    files = []
    for root, dirnames, filenames in os.walk(input_path):
        for filename in fnmatch.filter(filenames, '*.py'):
            files.append(os.path.join(root, filename))
    print(files)

    local_paths = {}
    with tqdm(total=len(files), desc="", leave=True) as pbar:
        for filepath in files:
            pbar.set_description(f"Processing {filepath}")
            
            local_path = filepath[len(input_path)+1:-3]
            grph = process_file(filepath, verbose=verbose)

            for name, edges, node in grph:
                local_paths[name] = local_path
                graph.append((
                        name,
                        edges,
                        visitors.to_source(node).replace('\\', '').strip()
                    ))
            pbar.update(1)

    def print_graph(gr):
        gr = list(map(lambda x: x[:2], gr))
        print(*gr, sep='\n')


    def map_local_name(func_name):
        if func_name in local_paths:
            return f"{local_paths[func_name]}.{func_name}"
        else:
            return f"<unk>.{func_name}"
    grph = []

    for name, edges, node in graph:
        grph.append((
            map_local_name(name),
            list(map(map_local_name, edges)),
            node
        ))

    graph = grph

    unk_counter = 0
    total_counter = 0
    for name, edges, node in graph:
        total_counter += len(edges)

        indicator = lambda x: x.startswith("<unk>")
        unk_counter += sum(map(indicator, edges))

    file_utils.save_graph_edges(graph, edges_path)
    file_utils.save_graph_nodes(graph, node_path)

    print(f"Unks: {unk_counter}/{total_counter}")