import pandas as pd
from astmonkey import visitors

def save_graph_edges(svace_graph, path):
	arr = []
	for entry in svace_graph:
		for callee in entry[1]:
			arr.append((entry[0], callee, 0))
	df = pd.DataFrame(arr, columns=['node1', 'node2', 'type'])
	df.to_csv(path, index=False)

def save_graph_nodes(vertices, path):

	df = pd.DataFrame(vertices)
	print("Saving dataframe with columns:", df.columns, "size:", len(df), "to", path)
	df.to_csv(path, index=False)
