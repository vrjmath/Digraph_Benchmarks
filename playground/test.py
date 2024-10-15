import programl as pg
from programl.proto import ProgramGraph
from programl.proto import ProgramGraphFeaturesList
from programl.util.py import pbutil
import pathlib
import networkx as nx
graph_def = ProgramGraphFeaturesList()
graph_def = ProgramGraph()
GRAPH_PB_PATH = './poj104_64.37617.1.ProgramGraph.pb'
GRAPH_PB_PATH = '/usr/scratch/vshitole6/Fall/Downstream/devmap/graphs_nvidia/'
file_name = 'shoc-1.1.5-S3D-ratx4_kernel-default.ProgramGraph.pb'
file_name = 'shoc-1.1.5-S3D-ratt_kernel-default.ProgramGraph.pb'
GRAPH_PB_PATH = GRAPH_PB_PATH + file_name
#GRAPH_PB_PATH = GRAPH_PB_PATH + 'labels/reachability/poj104_46.1565.2.ProgramGraphFeaturesList.pb'
#GRAPH_PB_PATH = GRAPH_PB_PATH + 'graphs/tensorflow.71875.cc.ProgramGraph.pb'
#GRAPH_PB_PATH = '/usr/scratch/vshitole6/Summer/PGML_Test/dataflow/logs/programl/reachability/ddf_30/checkpoints/001.Checkpoint.pb'
pth = pathlib.Path(GRAPH_PB_PATH)
graph = pbutil.FromFile(pth, graph_def)
print(pbutil.ToJson(graph))
gg = pg.to_networkx(graph)
#print(gg)
#pg.save_graphs('in.txt', [graph])
#with open('in.txt', 'w') as output:
 #   output.write(graph)
#print(list(gg.nodes(data=True))[0])
#print(list(gg.nodes(data=True))[1])
#print(list(gg.nodes(data=True))[2])
#print(list(gg.nodes(data=True))[3])
#print(nx.is_directed(gg))
#print(nx.degree_histogram(gg))
