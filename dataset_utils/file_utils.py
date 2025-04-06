import pandas as pd
from astmonkey import visitors

def save_graph_edges(graph, path):
	arr = []
	for name, edges, source_code in graph:
		for name2 in edges:
			print(name, name2)
			arr.append((name, name2))
	df = pd.DataFrame(arr, columns=['node1', 'node2'])
	df.to_csv(path, index=False)

def save_graph_nodes(graph, path):
	arr = []
	for name, edges, source_code in graph:
		arr.append((name, source_code))

	df = pd.DataFrame(arr, columns=['node', 'source_code'])
	df.to_csv(path, index=False)
