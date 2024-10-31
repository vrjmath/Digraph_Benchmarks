from pathlib import Path
import wandb
import networkx as nx
import torch_geometric
import torch
import random
import time
import os
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import convert

from models import GCN_Autoencoder, Graph_Prediction
from models import BidirectionalSAGE, DirectionalSAGE, BidirectionalGIN, DirectionalGIN
from data import load_pickle, read_vocabulary
from util import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch_geometric.utils import dense_to_sparse
import numpy as np
import pickle


wandb_log = False
table = []

def train_loop(model, criterion, optimizer, device, train_loader, validation_loader, epochs, patience, undirected):
    best_val_loss = float('inf')
    best_val_f1 = float('inf')
    best_val_roc = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    global table, wandb_log
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        output_true = []
        output_pred = []
        for data in train_loader:
            optimizer.zero_grad()
            if undirected == True:
                reverse_edge_index = data.edge_index[[1, 0], :]
                reverse_edge_index = torch.cat([data.edge_index, reverse_edge_index], dim=1)
                out = model(data.embedding.to(device), reverse_edge_index.to(device), data.batch.to(device))
            else:
                out = model(data.embedding.to(device), data.edge_index.to(device), data.batch.to(device))
            real = torch.FloatTensor(data.label).to(device)
            output_pred.append(out)
            output_true.append(real)
            loss = criterion(out, real)
            loss.backward(retain_graph=True)    
            optimizer.step()
            total_loss = loss + total_loss
        output_true = torch.cat(output_true)
        output_pred = torch.cat(output_pred)

        f1 = f1_score(output_true, output_pred)
        roc_auc = roc_auc_score(output_true.cpu(), torch.sigmoid(output_pred).detach().cpu())
        if wandb_log:
            wandb.log({'Train BCE loss': total_loss, 'Train F1 score': f1, 'Train ROC-AUC': roc_auc}, step=epoch)
        print(f"Epoch {epoch}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
        
        model.eval()
        total_loss = 0
        output_true = []
        output_pred = []
        for data in validation_loader:
            if undirected == True:
                reverse_edge_index = data.edge_index[[1, 0], :]
                reverse_edge_index = torch.cat([data.edge_index, reverse_edge_index], dim=1)
                out = model(data.embedding.to(device), reverse_edge_index.to(device), data.batch.to(device))
            else:
                out = model(data.embedding.to(device), data.edge_index.to(device), data.batch.to(device))
            real = torch.FloatTensor(data.label).to(device)
            output_pred.append(out)
            output_true.append(real)
            loss = criterion(out, real)
            total_loss = loss + total_loss
        output_true = torch.cat(output_true)
        output_pred = torch.cat(output_pred)
        f1 = f1_score(output_true, output_pred)
        roc_auc = roc_auc_score(output_true.cpu(), torch.sigmoid(output_pred).detach().cpu())
        print(f"Validation F1 Score: {f1:.4f}, Validation ROC AUC: {roc_auc:.4f}")
        if wandb_log:
            wandb.log({'Val BCE loss': total_loss, 'Val F1 score': f1, 'Val ROC-AUC': roc_auc}, step=epoch)

        if total_loss < best_val_loss:
            best_val_loss = total_loss
            best_val_f1 = f1
            best_val_roc = roc_auc
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    table.extend([best_val_f1, best_val_roc])
    return best_model_state

def test_loop(model, device, loader, undirected):
    global table, wandb_log
    model.eval()
    output_true = []
    output_pred = []
    for data in loader:
        if undirected == True:
            reverse_edge_index = data.edge_index[[1, 0], :]
            reverse_edge_index = torch.cat([data.edge_index, reverse_edge_index], dim=1)
            out = model(data.embedding.to(device), reverse_edge_index.to(device), data.batch.to(device))
        else:
            out = model(data.embedding.to(device), data.edge_index.to(device), data.batch.to(device))
        real = torch.FloatTensor(data.label).to(device)
        output_true.append(real)
        output_pred.append(out)
    output_true = torch.cat(output_true)
    output_pred = torch.cat(output_pred)
    f1 = f1_score(output_true, output_pred)
    roc_auc = roc_auc_score(output_true.cpu(), torch.sigmoid(output_pred).detach().cpu())
    print(f"Test F1 Score: {f1:.4f}, Test ROC AUC: {roc_auc:.4f}")
    table.extend([f1, roc_auc])

def main(args):
    global table, wandb_log
    torch.set_num_threads(args.num_threads)
    """
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed)
    """
    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())

    if wandb_log:
        wandb.init(project=f'Downstream', name=f'{ts}')

    interval = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"The device being used is a {device}.")
    
    """
    vocab_csv = "vocab/programl.csv"
    vocab = read_vocabulary(vocab_csv)
    data_list, labels = load_pickle(vocab)
    print(f"Data loading time: {(time.time() - interval):.2f}")
    """
    with open('data/data_list.pkl', 'rb') as f:
        data_list = pickle.load(f)
    with open('data/labels.pkl', 'rb') as f:
        labels = pickle.load(f)

    interval = time.time()
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
    labels = np.array(labels)
    for fold, (train_idx, test_idx) in enumerate(skf.split(data_list, labels)):
        if fold > 0:
            wandb_log = False
        table.extend([args.num_threads, args.batch_size, args.epochs, args.patience, args.undirected, args.bidirectional, args.GNN, args.num_layers, args.seed])

        train_labels = labels[train_idx]
        test_data = [data_list[i] for i in test_idx]
        skf_val = StratifiedKFold(n_splits=9, shuffle=True, random_state=args.seed)
        for val_train_idx, val_test_idx in skf_val.split(train_idx, train_labels):
            train_data = [data_list[i] for i in train_idx[val_train_idx]]
            val_data = [data_list[i] for i in train_idx[val_test_idx]]
            break  

        train_loader = DataLoader(train_data, batch_size=args.batch_size)
        validation_loader = DataLoader(val_data, batch_size=args.batch_size)
        test_loader = DataLoader(test_data, batch_size=args.batch_size)
        print(f"Fold {fold + 1}: Train={len(train_data)}, Validation={len(val_data)}, Test={len(test_data)}")

        gnn_model = None
        if args.GNN == "SAGE" and args.bidirectional == True:
            gnn_model = BidirectionalSAGE(in_channels=32, hidden_channels=16, out_channels=32, num_layers=args.num_layers)
        elif args.GNN == "GIN" and args.bidirectional == True:
            gnn_model = BidirectionalGIN(in_channels=32, hidden_channels=16, out_channels=32, num_layers=args.num_layers)
        elif args.GNN == "SAGE" and args.bidirectional == False:
            gnn_model = DirectionalSAGE(in_channels=32, hidden_channels=16, out_channels=32, num_layers=args.num_layers)
        elif args.GNN == "GIN" and args.bidirectional == False:
            gnn_model = DirectionalGIN(in_channels=32, hidden_channels=16, out_channels=32, num_layers=args.num_layers)

        model = None
        optimizer = None
        if args.fine_tuning == True:
            print("This is a fine-tuning experiment on the downstream dataset.")
            autoencoder_model = GCN_Autoencoder(gnn_model=gnn_model, in_channels=32, hidden_channels=16, out_channels=32, num_layers=args.num_layers).to(device)
            autoencoder_model.load_state_dict(torch.load(args.pretrained_model, map_location=torch.device('cpu')))
            encoder_weights = autoencoder_model.encoder.state_dict()
            autoencoder_model.to(device)

            model = Graph_Prediction(gnn_model=gnn_model, in_channels=32, hidden_channels=16, out_channels=1, num_layers=args.num_layers).to(device)
            model.encoder.load_state_dict(encoder_weights)
            for param in model.encoder.parameters():
                param.requires_grad = False
            optimizer = torch.optim.Adam(model.decoder.parameters(), lr=0.01)
            #optimizer = torch.optim.Adam(list(autoencoder_model.encoder.parameters()) + list(model.decoder.parameters()), lr=0.01)
        
        elif args.fine_tuning == False:
            print("This is a downstream experiment only.")
            model = Graph_Prediction(gnn_model=gnn_model, in_channels=32, hidden_channels=16, out_channels=1, num_layers=args.num_layers).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        criterion = torch.nn.BCEWithLogitsLoss()
        best_model_state = train_loop(model, criterion, optimizer, device, train_loader, validation_loader, epochs=args.epochs, patience=args.patience, undirected=args.undirected)
        model.load_state_dict(best_model_state)
        test_loop(model, device, test_loader, undirected=args.undirected)
    
        table.append(ts)
        with open(args.log_file_name, 'a') as f:
            if (os.stat(args.log_file_name).st_size == 0):
                columns = ["Num Threads", "Batch Size", "Epochs", "Patience", "Undirected", "Bidirectional", "GNN", "# Layers", "seed", "F1 Val", "ROC Val", "F1 Test", "ROC Test", "ts"]
                f.write(" ".join(f"{title:<14}" for title in columns) + '\n')
            f.write(" ".join(f"{value:<14.3f}" if isinstance(value, float) else f"{value:<14}" for value in table) + '\n')
        print(f"Training and testing time: {(time.time() - interval):.2f}")
        table = []
        if wandb_log: 
            wandb.finish()

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--num_threads", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--undirected", action='store_true', default=False)
    parser.add_argument("--bidirectional", action='store_true', default=False)
    parser.add_argument("--wandb", action='store_true', default=False)
    parser.add_argument("--fine_tuning", action='store_true', default=False)
    parser.add_argument("--GNN", type=str, default="SAGE")
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--log_file_name", type=str, default="default.txt")
    parser.add_argument("--pretrained_model", type=str, default="default.txt")
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()
    wandb_log = args.wandb
    main(args)