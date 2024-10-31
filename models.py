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
    def __init__(self, in_channels, out_channels):
        super(FinalLayer, self).__init__()
        self.lin = Linear(in_channels, out_channels)

    def forward(self, x, batch):
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x
    
class MLPClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MLPClassifier, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels))
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.mlp(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

class GCNDecoder(nn.Module):
    def __init__(self, out_channels):
        super(GCNDecoder, self).__init__()
        self.out_channels = out_channels

    def forward(self, z):
        # Reconstruct the adjacency matrix (simplified inner product decoder)
        adj_pred = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_pred
     
class EdgeTypeDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(EdgeTypeDecoder, self).__init__()
        self.fc1 = nn.Linear(2 * in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, 1)
        self.bilinear = nn.Bilinear(in_channels, in_channels, 1)  # Bilinear layer
    
    def forward(self, z, edge_index):
        src, dst = edge_index
        z_src = z[src]
        z_dst = z[dst]
        out = self.bilinear(z_src, z_dst)
        out = torch.sigmoid(out).squeeze()
        return out
    
class GCN_Autoencoder(nn.Module):
    def __init__(self, gnn_model, in_channels, hidden_channels, out_channels, num_layers):
        super(GCN_Autoencoder, self).__init__()
        self.encoder = gnn_model
        self.decoder = EdgeTypeDecoder(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, batch):
        z = self.encoder(x, edge_index, batch)
        adj_pred = self.decoder(z, edge_index)
        return adj_pred

class Graph_Prediction(nn.Module):
    def __init__(self, gnn_model, in_channels, hidden_channels, out_channels, num_layers):
        super(Graph_Prediction, self).__init__()
        self.encoder = gnn_model
        self.decoder = FinalLayer(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, batch):
        z = self.encoder(x, edge_index, batch)
        graph_pred = self.decoder(z, batch)
        return graph_pred



class BidirectionalGINConv(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels))
        self.conv_forward = GINConv(mlp)
        self.conv_backward = GINConv(mlp)

    def forward(self, x, edge_index, reverse_edge_index):
        x_forward = self.conv_forward(x, edge_index)
        x_backward = self.conv_backward(x, reverse_edge_index)
        x = (x_forward + x_backward)/2
        x = F.relu(x)
        return x
    
class BidirectionalGIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(BidirectionalGIN, self).__init__()
        
        self.convs = nn.ModuleList()
        assert num_layers >= 1
        self.convs.append(BidirectionalGINConv(in_channels, hidden_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(BidirectionalGINConv(hidden_channels, hidden_channels, hidden_channels))
        #self.lin = nn.Linear(hidden_channels, out_channels)
    def forward(self, x, edge_index, batch):
        reverse_edge_index = edge_index[[1, 0], :]
        for conv in self.convs:
            x = conv(x, edge_index, reverse_edge_index)

        #x = global_mean_pool(x, batch)
        #x = self.lin(x)

        return x

class DirectionalGINConv(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout())
        """
        mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels))"""
        self.conv_forward = GINConv(mlp)

    def forward(self, x, edge_index):
        x = self.conv_forward(x, edge_index)
        x = F.relu(x)
        return x
    
class DirectionalGIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(DirectionalGIN, self).__init__()
        
        self.convs = nn.ModuleList()
        assert num_layers >= 1
        self.convs.append(DirectionalGINConv(in_channels, hidden_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(DirectionalGINConv(hidden_channels, hidden_channels, hidden_channels))
        #self.lin = nn.Linear(hidden_channels, out_channels)
    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)

        #x = global_mean_pool(x, batch)
        #x = self.lin(x)

        return x
    
class BidirectionalSAGEConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_forward = SAGEConv(in_channels, out_channels)
        self.conv_backward = SAGEConv(in_channels, out_channels)

    def forward(self, x, edge_index, reverse_edge_index):
        x_forward = self.conv_forward(x, edge_index)
        x_backward = self.conv_backward(x, reverse_edge_index)
        x = (x_forward + x_backward)/2
        x = F.relu(x)
        return x
    
class BidirectionalSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(BidirectionalSAGE, self).__init__()
        
        self.convs = nn.ModuleList()
        assert num_layers >= 1
        self.convs.append(BidirectionalSAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(BidirectionalSAGEConv(hidden_channels, hidden_channels))
        #self.lin = nn.Linear(hidden_channels, out_channels)
    def forward(self, x, edge_index, batch):
        reverse_edge_index = edge_index[[1, 0], :]
        for conv in self.convs:
            x = conv(x, edge_index, reverse_edge_index)
            
        #x = global_mean_pool(x, batch)
        #x = self.lin(x)
        return x
    
class DirectionalSAGEConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_forward = SAGEConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv_forward(x, edge_index)
        x = F.relu(x)
        return x
    
class DirectionalSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(DirectionalSAGE, self).__init__()
        
        self.convs = nn.ModuleList()
        assert num_layers >= 1
        self.convs.append(DirectionalSAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(DirectionalSAGEConv(hidden_channels, hidden_channels))
        #self.lin = nn.Linear(hidden_channels, out_channels)
    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            
        #x = global_mean_pool(x, batch)
        #x = self.lin(x)
        return x
    
