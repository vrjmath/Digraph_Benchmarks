# 📘 README

## Overview

This project benchmarks **graph neural networks (GNNs)** on the OpenCL DevMap dataset.
It supports multiple GNNs, bidirectional message passing, and directed/undirected graphs. 

Key features:

* 10-fold stratified cross-validation
* Training with early stopping
* Support for **fine-tuning** pretrained autoencoders
* Logging to **Weights & Biases (wandb)**
* Evaluation with **F1-score** and **ROC-AUC**

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
