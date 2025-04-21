# import dataset_utils.file_utils
from dataset_utils import ast_utils, file_utils

import ast
from astmonkey import visitors
from tqdm import tqdm
import os

from dataset_utils.ast_utils import get_vertices
from dataset_utils.svace_utils import get_svace_graph

def rename_svace_graph(vertices, svace_graph):
    svace_vertices = []
    for entry in svace_graph:
        svace_vertices.append(entry['function'])
        map(svace_vertices.append, entry['callees'])
    svace_vertices = list(set(svace_vertices))
    print(svace_vertices[:5])

    print()

    print(vertices[:5])

    short2normal = dict()
    for func_name, _, _ in vertices:
        i = func_name.rfind('/')
        short2normal[func_name[i+1:]] = func_name

    def rename_svace_vertex(name):
        # name = 'f5d48c8d5cfdfd5f5762c34dac4a4e0795682ce4.constants.py:<module>:TableFormat'

        parts = list(name.split(":")) # ['f5d48c8d5cfdfd5f5762c34dac4a4e0795682ce4.constants.py', '<module>', 'TableFormat']

        parts[0] = list(parts[0].split('.'))[-2] # constants

        assert "<module>" in parts
        parts.remove("<module>")

        func_name = name[name.rfind(":")+1:]
        
        guess1 = '.'.join(parts)
        if guess1 in short2normal:
            return short2normal[guess1]
        
        # print(name, end='\n\n')
        return None

    cringe2normal = dict(zip(
        svace_vertices,
        map(rename_svace_vertex, svace_vertices)
    ))

    def rename_svace_vertex_safe(name):
        return cringe2normal.get(name, None)

    if True:
        count_none = sum(i is None for i in cringe2normal.values())
        print("Nan count: ", count_none, "out of", len(svace_vertices))

    new_svace_graph = []
    for entry in svace_graph:
        function = rename_svace_vertex_safe(entry['function'])

        if function is None:
            continue
        callees = list(map(rename_svace_vertex_safe, entry['callees']))
        print("Yesss", len(callees))
        callees = list(filter(lambda x: x is not None, callees))

        if len(callees) == 0:
            continue
        print("Yesss")
        new_svace_graph.append([function, callees])

    print(new_svace_graph[:5])
    raise NotImplementedError("Need to merge different jsons")
    return graph

def create_dataset(input_path, output_path=None, verbose=False):
    if output_path is None:
        output_path = input_path

    edges_path = f"{output_path}/edges.csv"
    node_path = f"{output_path}/nodes.csv"


    vertices = get_vertices(input_path)
    svace_graph = get_svace_graph(input_path)
    svace_graph = rename_svace_graph(vertices=vertices, svace_graph=svace_graph)

    save_graph_edges(svace_graph, edges_path)
    save_graph_nodes(vertices, node_path)


def create_multiple_datasets(input_path, output_path=None, verbose=False):
    for folder in tqdm(os.listdir(input_path), desc="Listing datasets for loading"):
        cur_folder = os.path.join(input_path, folder)
        if output_path is None:
            create_dataset(cur_folder, verbose=verbose)
        else:
            create_dataset(cur_folder, output_path=os.path.join(output_path, folder), verbose=verbose)