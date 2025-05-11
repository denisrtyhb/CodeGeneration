# import dataset_utils.file_utils
from codegen.dataset_utils import ast_utils, file_utils

import ast
from astmonkey import visitors
from tqdm import tqdm
import os


from codegen.dataset_utils.ast_utils import get_vertices
from codegen.dataset_utils.svace_utils import get_svace_graph
from codegen.dataset_utils.file_utils import save_graph_edges, save_graph_nodes

def filter_garbage(svace_vertices: list[str]) -> list[str]:
    
    def not_init(vertice: str) -> bool:
        i = vertice.rfind(":")
        last = vertice[i+1:]
        return not vertice.startswith("$$") and not vertice.endswith("$$")
    
    def not_lambda(vertice: str) -> bool:
        i = vertice.rfind(":")
        last = vertice[i+1:]
        return last != "<lambda>"

    def not_module(vertice: str) -> bool:
        i = vertice.rfind(":")
        last = vertice[i+1:]
        return last != "<module>"
    
    
    def not_2(vertice: str) -> bool:
        i = vertice.rfind(":")
        last = vertice[i+1:]
        return last != "2"
    svace_vertices = filter(not_init, svace_vertices)
    svace_vertices = filter(not_lambda, svace_vertices)
    svace_vertices = filter(not_module, svace_vertices)


    # TODO: Maybe it's not garbage. It's most confusing one
    # 2e8ab3d43f274579bb6c6.page.py:<module>:Page:signatures:2 <- ????
    svace_vertices = filter(not_2, svace_vertices)

    return list(svace_vertices)


def rename_svace_graph(svace_graph):
    verbose = False

    svace_vertices = []
    for entry in svace_graph:
        svace_vertices.append(entry['function'])
        map(svace_vertices.append, entry['callees'])
    svace_vertices = list(set(svace_vertices))
    
    if verbose:
        print("Vertices before", *svace_vertices[:5], sep='\n')

    name2path = dict()
    for entry in svace_graph:
        name = entry['function']
        path = entry['path']

        true_name = name[name.find("<module>"):]
        true_name = true_name[true_name.find(":")+1:]
        true_name = true_name.replace(":", ".")

        name2path[name] = path[:-2] + true_name
    
    new_svace_graph = []
    for entry in svace_graph:
        function = name2path[entry['function']]

        # TODO: fix the cases where name2path does not have item
        # Idk why that happens
        callees = list(map(lambda x: name2path.get(x, None), entry['callees']))
        callees = list(filter(lambda x: x is not None, callees))

        # callees = list(filter(lambda x: x is not None, callees))

        if len(callees) == 0:
            continue
        # print("Yesss")
        new_svace_graph.append([function, callees])
    
    if verbose:
        print("Vertices after", *new_svace_graph[:5], sep='\n')

    return new_svace_graph

def get_svace_vertices(svace_graph):
    all_svace_vertices = []
    for i in svace_graph:
        all_svace_vertices.append(i[0])
        for j in i[1]:
            all_svace_vertices.append(j)
    all_svace_vertices = list(set(all_svace_vertices))
    return all_svace_vertices

def check_mapping_completeness(vertices, svace_graph):
    
    all_svace_vertices = get_svace_vertices(svace_graph)
    all_vertices = list(map(lambda x: x["node"], vertices))


    print("Total svace vertices:", len(all_svace_vertices))
    print("Total vertices:", len(all_vertices))

    cnt = 0
    for i in all_svace_vertices:
        cnt += (i in all_vertices)
    print("Intersection:", cnt)
    print("Edges number:",
          sum(map(lambda x: len(x[1]), svace_graph)))


def create_dataset(input_path, output_path=None, verbose=False):
    print("Current dataset:", input_path)
    if output_path is None:
        output_path = input_path

    edges_path = f"{output_path}/edges.csv"
    node_path = f"{output_path}/nodes.csv"

    try:
        vertices = get_vertices(input_path)
        svace_graph = get_svace_graph(input_path)
        svace_graph = rename_svace_graph(svace_graph)


        check_mapping_completeness(vertices, svace_graph)
        svace_vertices = get_svace_vertices(svace_graph)

        vertices = filter(lambda x: x["node"] in svace_vertices, vertices)

        save_graph_edges(svace_graph, edges_path)
        save_graph_nodes(vertices, node_path)
    except Exception as e:
        print(f"Error processing dataset {input_path}: {e}")


def create_multiple_datasets(input_path, output_path=None, verbose=False):
    for folder in tqdm(os.listdir(input_path), desc="Listing datasets for loading"):
        cur_folder = os.path.join(input_path, folder)
        if output_path is None:
            create_dataset(cur_folder, verbose=verbose)
        else:
            create_dataset(cur_folder, output_path=os.path.join(output_path, folder), verbose=verbose)