# import os
# os.chdir("..")



import sys
print(sys.path)
sys.path.append("..")

import ast
from astmonkey import visitors

import ast_utils

if len(sys.argv) < 2:
    print("Usage: python script.py <filename>")
    sys.exit(1)

filename = sys.argv[1]
ast_tree = ast_utils.construct_ast_from_file(filename)
assert ast_tree, "Failed to construct AST."

visitor = ast_utils.GraphMakerVisitor()

graph = visitor.construct_graph(ast_tree)


if False:
	for name, edges, node in graph:
		source = visitors.to_source(node).replace('\\', '').strip()
		print("\n\n\n")
		print("Source code:")
		print(source)
		print("Edges:")
		print(name, "->", ', '.join(edges))


edges_path = "data/edges.csv"
node_path = "data/nodes.csv"

import file_utils

file_utils.save_graph_edges(graph, edges_path)
file_utils.save_graph_nodes(graph, node_path)