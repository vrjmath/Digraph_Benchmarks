import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import math
import torch
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GATConv, GINConv, global_add_pool, SAGEConv

from torch.nn import Linear

class GraphConvolution(Module):
    #https://arxiv.org/abs/1609.02907

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.lin = Linear(nclass, 1)
        self.lin2 = Linear(32, nclass)
        self.dropout = dropout

    def forward(self, x, adj, batch):
        
        x = self.gc1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.log_softmax(x, dim=1)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        
        """
        x = F.relu(self.lin2(x))
        x = F.log_softmax(x, dim=1)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        """
        return x

class GAT(torch.nn.Module):
    #https://arxiv.org/pdf/1710.10903
    def __init__(self, input_dim, hidden_dim, output_dim, heads=1):
        super(GAT, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)  # First GAT layer
        self.gat2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False)  # Second GAT layer
        self.lin = Linear(output_dim, 1)

    def forward(self, x, adj, batch):
        x = self.gat1(x, adj)
        x = F.relu(x)
        x = self.gat2(x, adj)
        x = F.log_softmax(x, dim=1)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x
    

class GCNEncoder(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCNEncoder, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.lin = Linear(nclass, 1)
        self.lin2 = Linear(32, nclass)
        self.dropout = dropout

    def forward(self, x, adj, batch):
        x = self.gc1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        #x = F.log_softmax(x, dim=1)
        #x = global_mean_pool(x, batch)
        #x = self.lin(x)

        """
        x = F.relu(self.lin2(x))
        x = F.log_softmax(x, dim=1)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        """
        return x
    
class FinalLayer(nn.Module):
    def __init__(self):
        super(FinalLayer, self).__init__()
        
        self.lin = Linear(5, 1)

    def forward(self, x, batch):
        x = F.log_softmax(x, dim=1)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x
    
class MLPClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class GCNDecoder(nn.Module):
    def __init__(self, out_channels):
        super(GCNDecoder, self).__init__()
        self.out_channels = out_channels

    def forward(self, z):
        # Reconstruct the adjacency matrix (simplified inner product decoder)
        adj_pred = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_pred
    
class MLPDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels=32):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(2 * in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, 1)
    
    def forward(self, z, edge_index):
        # edge_index has shape [2, E], where E is the number of edges to predict
        src, dst = edge_index
        # Gather the embeddings for source and target nodes
        z_src = z[src]  # Shape: [E, in_channels]
        z_dst = z[dst]  # Shape: [E, in_channels]
        # Concatenate embeddings
        z_pair = torch.cat([z_src, z_dst], dim=1)  # Shape: [E, 2 * in_channels]
        # Pass through MLP
        hidden = F.relu(self.fc1(z_pair))  # Shape: [E, hidden_channels]
        out = torch.sigmoid(self.fc2(hidden)).squeeze()  # Shape: [E]
        return out
    
class GCN_Autoencoder(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_Autoencoder, self).__init__()
        #self.encoder = GCNEncoder(nfeat, nhid, nclass, dropout)
        #self.encoder = DirectedGINLayer(nfeat, nclass)
        self.encoder = BidirectionalSAGEConv(nfeat, nhid, nclass)
        #self.decoder = GCNDecoder(nclass)
        self.decoder = MLPDecoder(5)

    def forward(self, x, adj, batch, edge_index):
        # Encoder: Get node embeddings
        z = self.encoder(x, adj, batch)
        # Decoder: Reconstruct adjacency matrix
        #adj_pred = self.decoder(z)
        adj_pred = self.decoder(z, edge_index)
        return adj_pred


class BidirectionalGINConv(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(BidirectionalGINConv, self).__init__()
        mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels))
        self.gin_forward = GINConv(mlp)
        self.gin_backward = GINConv(mlp)
        self.lin = Linear(out_channels, 1)

    def forward(self, x, edge_index, batch):
        x_forward = self.gin_forward(x, edge_index)
        reverse_edge_index = edge_index[[1, 0], :]
        x_backward = self.gin_backward(x, reverse_edge_index)
        x_bidirectional = (x_forward + x_backward) / 2.0
        x = F.log_softmax(x_bidirectional, dim=1)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

class BidirectionalSAGEConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        
        self.conv_forward = SAGEConv(in_channels, out_channels)
        self.conv_backward = SAGEConv(in_channels, out_channels)
    
    def forward(self, x, edge_index, reverse_edge_index):
        x1 = self.conv_forward(x, edge_index)
        x2 = self.conv_backward(x, reverse_edge_index)
        x = (x1 + x2)/2
        x = F.relu(x)
        
        return x
    
class BidirectionalSAGE(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers):
        super().__init__()

        self.convs = nn.ModuleList()
        assert num_layers >= 1
        self.convs.append(BidirectionalSAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(BidirectionalSAGEConv(hidden_channels, hidden_channels))
        self.pred = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, batch):
        reverse_edge_index = edge_index[[1, 0], :]
        for conv in self.convs:
            x = conv(x, edge_index, reverse_edge_index)
        x = global_mean_pool(x, batch)
        x = self.pred(x)
        
        return x
