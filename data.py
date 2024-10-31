import pickle
import torch_geometric
import glob
import torch
import csv
from torch import nn
from tqdm import tqdm
from deepdataflow import DeepDataFlow 

def obtain_embeddings(vocab, G):
    arr = []
    for i in range(len(list(G.nodes(data=True)))):
        if (vocab.get(list(G.nodes(data=True))[i][1]["text"])) != None:
            arr.append(vocab[list(G.nodes(data=True))[i][1]["text"]])
        else:
            arr.append(vocab["None"])
    arr = torch.squeeze(torch.stack(arr))
    return arr

def load_pickle(vocab):
    data_list = []
    labels = []
    files = [file for file in glob.glob("program_graphs/*")]
    i = 0
    for file_name in tqdm(files):
        #if i > 100:
            #break
        i = i + 1
        G = pickle.load(open(file_name, 'rb'))
        for node in G.nodes:
            if 'features' in G.nodes[node]:
                del G.nodes[node]['features']        
        try: 
            torch_data = torch_geometric.utils.from_networkx(G)
            torch_data.features["embedding"] = obtain_embeddings(vocab, G)
            torch_deepdataflow = DeepDataFlow(edge_index=torch_data.edge_index, type=torch_data.type, flow=torch_data.flow, 
                         position = torch_data.position, embedding = obtain_embeddings(vocab, G), 
                         label = torch_data.features["devmap_label"], num_nodes = torch_data.num_nodes)
            data_list.append(torch_deepdataflow)
            labels.append(torch_data.features["devmap_label"])
        except ValueError:
            print("This graph contains different node attributes")
    return data_list, labels

def read_vocabulary(vocab_csv):
    target_cumfreq = 1.0
    max_items = 0

    vocab = {}
    cumfreq = 0
    with open(vocab_csv) as f:
        vocab_file = csv.reader(f.readlines(), delimiter="\t")
        for i, row in enumerate(vocab_file, start=-1):
            if i == -1:  
                continue
            (cumfreq, _, _, text) = row
            cumfreq = float(cumfreq)
            vocab[text] = i
            if cumfreq >= target_cumfreq:
                break
            if max_items and i >= max_items - 1:
                break
    vocab["None"] = 2230
    embedding_shape = (len(vocab) + 1, 32)
    node_embeddings = nn.Embedding(*embedding_shape)

    for x in vocab:
        vocab[x] = node_embeddings(torch.LongTensor([vocab[x]]))

    return vocab