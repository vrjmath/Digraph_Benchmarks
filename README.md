# 📘 README

## Overview

This project benchmarks **graph neural networks (GNNs)** on a downstream classification task.
It supports training from scratch or fine-tuning from a pretrained **GCN Autoencoder**, with options for different GNN backbones (GraphSAGE, GIN), directional vs. bidirectional message passing, and undirected graph treatment.

Key features:

* 10-fold stratified cross-validation
* Training with early stopping
* Support for **fine-tuning** pretrained autoencoders
* Logging to **Weights & Biases (wandb)**
* Evaluation with **F1-score** and **ROC-AUC**

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
```

Make sure you have PyTorch Geometric installed. Follow the official [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for your system.

---

## 📂 Data

The training script expects:

* `data/data_list.pkl` → list of graphs (`torch_geometric.data.Data` objects with `.embedding`, `.edge_index`, `.label`)
* `data/labels.pkl` → corresponding graph labels

To customize dataset loading, modify `main()` where these files are read.

---

## 🚀 Usage

### Training from scratch

```bash
python train.py \
  --num_threads 4 \
  --batch_size 32 \
  --epochs 100 \
  --patience 10 \
  --GNN SAGE \
  --num_layers 2 \
  --log_file_name results.txt \
  --seed 42
```

### Fine-tuning from a pretrained autoencoder

```bash
python train.py \
  --num_threads 4 \
  --batch_size 32 \
  --epochs 50 \
  --patience 5 \
  --GNN GIN \
  --bidirectional \
  --fine_tuning \
  --pretrained_model pretrained/autoencoder.pth \
  --log_file_name finetune_results.txt \
  --seed 123
```

### With Weights & Biases logging

```bash
python train.py --wandb
```

---

## ⚙️ Arguments

| Argument             | Type | Default         | Description                                             |
| -------------------- | ---- | --------------- | ------------------------------------------------------- |
| `--num_threads`      | int  | `2`             | Number of CPU threads to use.                           |
| `--batch_size`       | int  | `2`             | Mini-batch size.                                        |
| `--epochs`           | int  | `10`            | Maximum number of epochs.                               |
| `--patience`         | int  | `10`            | Early stopping patience.                                |
| `--undirected`       | flag | `False`         | If set, treat graphs as undirected (add reverse edges). |
| `--bidirectional`    | flag | `False`         | If set, use bidirectional GNN variant.                  |
| `--wandb`            | flag | `False`         | Enable logging to Weights & Biases.                     |
| `--fine_tuning`      | flag | `False`         | Fine-tune a pretrained autoencoder.                     |
| `--GNN`              | str  | `"SAGE"`        | GNN backbone: `SAGE` or `GIN`.                          |
| `--num_layers`       | int  | `1`             | Number of GNN layers.                                   |
| `--log_file_name`    | str  | `"default.txt"` | File to log results.                                    |
| `--pretrained_model` | str  | `"default.txt"` | Path to pretrained model (if fine-tuning).              |
| `--seed`             | int  | `1`             | Random seed for reproducibility.                        |

---

## 📊 Output

* Prints **F1 score** and **ROC-AUC** for train, validation, and test sets per fold.
* Logs results in the specified `--log_file_name` with columns:

```
Num Threads | Batch Size | Epochs | Patience | Undirected | Bidirectional | GNN | # Layers | Seed | F1 Val | ROC Val | F1 Test | ROC Test | Timestamp
```

* If `--wandb` is enabled, results are also logged to your wandb dashboard.

---

## 🔧 Project Structure

```
.
├── data/
│   ├── data_list.pkl
│   ├── labels.pkl
├── models.py
├── util.py
├── train.py
├── requirements.txt
└── README.md
```

---

## 📝 Citation

If you use this code, please cite this repository.
