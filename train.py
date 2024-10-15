from pathlib import Path
import wandb
import networkx as nx
import torch_geometric
import torch
import random
import time
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import convert

from models import GCN, GAT, GCN_Autoencoder, FinalLayer, MLPClassifier, DirectedGINLayer, DirectedGraphSAGE, BidirectionalGINConv
from models import BidirectionalSAGEConv
from data import load_pickle, read_vocabulary
from util import f1_score
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import dense_to_sparse


def train_loop(model, criterion, optimizer, device, loader, epochs, test_loader):
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        output_true = []
        output_pred = []
        for data in loader:
            adj_matrix = torch_geometric.utils.convert.to_scipy_sparse_matrix(data.edge_index)
            optimizer.zero_grad()
            features = data.features["embedding"].to(device)
            out = model(features, data.edge_index.to(device), data.batch.to(device))
            #out = model(features, torch.from_numpy(adj_matrix.toarray()).to(device), data.batch.to(device))
            #out = model(features, dense_to_sparse(torch.from_numpy(adj_matrix.toarray()))[0].to(device), data.batch.to(device))

            p = torch.FloatTensor(data.features["devmap_label"]).to(device)
            output_true.append(p)
            output_pred.append(out)
            loss = criterion(out, p)
            loss.backward(retain_graph=True)    
            optimizer.step()
            total_loss = loss + total_loss
        output_true = torch.cat(output_true)
        output_pred = torch.cat(output_pred)

        f1 = f1_score(output_true, output_pred)
        roc_auc = roc_auc_score(output_true, torch.sigmoid(output_pred).detach())
        #wandb.log({'BCE loss': total_loss, 'F1 score': f1, 'ROC-AUC': roc_auc}, step=epoch)

        print(f"Epoch {epoch}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
        #if epoch%2 == 0:
            #test_loop(model, criterion, optimizer, device, test_loader)


def test_loop(model, criterion, optimizer, device, loader):
    model.eval()
    output_true = []
    output_pred = []
    for data in loader:
        adj_matrix = torch_geometric.utils.convert.to_scipy_sparse_matrix(data.edge_index)
        features = data.features["embedding"].to(device)
        out = model(features, data.edge_index.to(device), data.batch.to(device))
        #out = model(features, torch.from_numpy(adj_matrix.toarray()).to(device), data.batch.to(device))
        #out = model(features, dense_to_sparse(torch.from_numpy(adj_matrix.toarray()))[0].to(device), data.batch.to(device))
        p = torch.FloatTensor(data.features["devmap_label"]).to(device)
        output_true.append(p)
        output_pred.append(out)
    output_true = torch.cat(output_true)
    output_pred = torch.cat(output_pred)
    f1 = f1_score(output_true, output_pred)
    roc_auc = roc_auc_score(output_true, torch.sigmoid(output_pred).detach())
    print(f"Test F1 Score: {f1:.4f}, Test ROC AUC: {roc_auc:.4f}")
    #wandb.log({'Test F1 score': f1, 'Test ROC-AUC': roc_auc})


def downstream_test_loop(model, classifier, criterion, device, loader, loader2):
    final_layer = FinalLayer()
    optimizer = torch.optim.Adam(list(model.encoder.parameters()) + list(final_layer.parameters()), lr=0.01)
    for param in model.parameters():
        param.requires_grad = False 
    for epoch in range(100):
        total_loss = 0
        output_true = []
        output_pred = []
        for data in loader:
            optimizer.zero_grad()
            adj_matrix = torch_geometric.utils.convert.to_scipy_sparse_matrix(data.edge_index)
            features = data.features["embedding"].to(device)
            out = model.encoder(features, data.edge_index.to(device), data.batch.to(device))
            #out = model.encoder(features, torch.from_numpy(adj_matrix.toarray()).to(device), data.batch.to(device))
            out = final_layer(out, data.batch)
            p = torch.FloatTensor(data.features["devmap_label"]).to(device)
            output_true.append(p)
            output_pred.append(out)
            loss = criterion(out, p)
            loss.backward(retain_graph=True)    
            optimizer.step()
            total_loss = loss + total_loss
        output_true = torch.cat(output_true)
        output_pred = torch.cat(output_pred)
        #output_pred = torch.sigmoid(output_pred).detach()
        f1 = f1_score(output_true, output_pred)
        roc_auc = roc_auc_score(output_true, torch.sigmoid(output_pred).detach())
        #wandb.log({'BCE loss': total_loss, 'F1 score': f1, 'ROC-AUC': roc_auc}, step=epoch)
        print(f"Epoch {epoch}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

    output_true = []
    output_pred = []
    for data in loader2:
        adj_matrix = torch_geometric.utils.convert.to_scipy_sparse_matrix(data.edge_index)
        features = data.features["embedding"].to(device)
        out = model.encoder(features, data.edge_index.to(device), data.batch.to(device))
        #out = model.encoder(features, torch.from_numpy(adj_matrix.toarray()).to(device), data.batch.to(device))
        out = final_layer(out, data.batch)
        p = torch.FloatTensor(data.features["devmap_label"]).to(device)
        output_true.append(p)
        output_pred.append(out)
    output_true = torch.cat(output_true)
    output_pred = torch.cat(output_pred)
    f1 = f1_score(output_true, output_pred)
    roc_auc = roc_auc_score(output_true, torch.sigmoid(output_pred).detach())
    print(f"Test F1 Score: {f1:.4f}, Test ROC AUC: {roc_auc:.4f}")

def main(args):
    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    #seed = 42  # Choose any fixed seed
    #torch.manual_seed(seed)
    interval = time.time()
    """wandb.init(
        project=f'Downstream',
        name=f'{ts}',
    )"""

    vocab_csv = "vocab/programl.csv"
    vocab = read_vocabulary(vocab_csv)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"The device being used is a {device}.")

    data_list = load_pickle(vocab)
    #random.shuffle(data_list)
    split = int(0.8*len(data_list))
    train_data = data_list[0:split]
    test_data = data_list[split:len(data_list)]
    print(f"{len(train_data)} graphs in the train dataset.")
    print(f"{len(test_data)} graphs in the test dataset.")
    train_loader = DataLoader(train_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)
    
    print(f"Data loading time: {(time.time() - interval):.2f}")
    interval = time.time()

    criterion = torch.nn.BCEWithLogitsLoss()
    gcn_autoencoder = GCN_Autoencoder(nfeat=32,
            nhid=16,
            nclass=5,
            dropout=0.5).to(device)

    #gcn_autoencoder.load_state_dict(torch.load('vocab/autoencoder_gs_reconstruction_short.pth', map_location=torch.device('cpu')))
    #gcn_autoencoder.eval()
    """
    model = GCN(nfeat=32,
            nhid=16,
            nclass=5,
            dropout=0.5).to(device)"""
    model = BidirectionalSAGEConv(in_channels=32, hidden_channels=16, out_channels=5)
    #model = DirectedGINLayer(in_channels=32, out_channels=5)
    #model = DirectedGraphSAGE(in_channels=32, out_channels=5)
    #classifier = MLPClassifier(input_dim=5, output_dim=1)
    #downstream_test_loop(gcn_autoencoder, classifier, criterion, device, train_loader, test_loader)
    
    #model2 = GAT(input_dim=32, hidden_dim=16, output_dim=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_loop(model, criterion, optimizer, device, train_loader, args.epochs, test_loader)
    test_loop(model, criterion, optimizer, device, test_loader)

    print(f"Training and testing time: {(time.time() - interval):.2f}")
    #wandb.finish()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    main(args)