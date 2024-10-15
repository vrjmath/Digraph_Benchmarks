import programl as pg
from programl.proto import ProgramGraph
from programl.proto import ProgramGraphFeaturesList
from programl.util.py import pbutil
import pathlib
from pathlib import Path
import networkx as nx
import glob
import pickle
import torch_geometric
files = [file for file in glob.glob('/usr/scratch/vshitole6/Fall/Downstream/devmap/graphs_nvidia/*')]
count = 1
for file_name in files:
	if count % 100 == 0:
		print(count)
	if count <= 0:
		graph = pbutil.FromFile(Path(file_name), ProgramGraph())
		nx_graph = pg.to_networkx(graph)
		pickle.dump(nx_graph, open('program_graphs/graph' + str(count) + '.pickle', 'wb'))
		count = count + 1

ggg = pickle.load(open('program_graphs/graph400.pickle', 'rb'))
print(ggg)
#torch_data = torch_geometric.utils.from_networkx(ggg)
#print(torch_data["graph_features"])

G = ggg
for node in G.nodes:
	if 'features' in G.nodes[node]:
		del G.nodes[node]['features']

print(list(G.nodes(data=True)))

torch_data = torch_geometric.utils.from_networkx(G)
print(torch_data)
