import torch
from torch_geometric.data import Data, Batch


class DeepDataFlow(Data):
    def __init__(self, edge_index=None, type = None, flow=None, position=None, embedding=None, label = None, num_nodes=None, **kwargs):
        super().__init__(edge_index=edge_index, **kwargs)
        
        self.edge_index = edge_index
        self.type = type
        self.flow = flow
        self.position = position
        self.embedding = embedding
        self.label = label
        self.num_nodes = num_nodes
