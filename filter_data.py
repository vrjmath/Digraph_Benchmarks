import torch
from data import load_pickle, read_vocabulary
import pickle


def main(args):
    """
    with open('data_list.pkl', 'rb') as f:
        data_list = pickle.load(f)
    print(data_list)
    """
    
    vocab_csv = "vocab/programl.csv"
    vocab = read_vocabulary(vocab_csv)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"The device being used is a {device}.")
    data_list, labels = load_pickle(vocab)

    with open('data/data_list.pkl', 'wb') as f:
        pickle.dump(data_list, f)

    with open ('data/labels.pkl', 'wb') as f:
        pickle.dump(labels, f)
    
    print("Data loading complete.")

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)