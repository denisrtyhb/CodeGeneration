import pandas as pd
from astmonkey import visitors

def save_graph_edges(graph, path):

	all_names = list(map(lambda x: x[0], graph))

	def identity_type(name1, name2):
		if name2 in all_names:
			return "dependent"
		elif name2.startswith("<unk>"):
			return "unknown"
		else:
			return "basic"

	arr = []
	for name, edges, source_code in graph:
		for name2 in edges:
			arr.append((name, name2, identity_type(name, name2)))
	df = pd.DataFrame(arr, columns=['node1', 'node2', 'type'])
	df.to_csv(path, index=False)

def save_graph_nodes(graph, path):
	arr = []
	for name, edges, source_code in graph:
		arr.append((name, source_code))

	df = pd.DataFrame(arr, columns=['node', 'source_code'])
	df.to_csv(path, index=False)
