# README

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

## Usage

```bash
python train.py --wandb
```


| Argument             | Type | Default         | Description                                             |
| -------------------- | ---- | --------------- | ------------------------------------------------------- |
| `--num_threads`      | int  | `2`             | Number of CPU threads to use.                           |
| `--batch_size`       | int  | `2`             |                                         |
| `--epochs`           | int  | `10`            |                                |
| `--patience`         | int  | `10`            | Early stopping patience.                                |
| `--undirected`       | flag | `False`         | If true, treat graphs as undirected (add reverse edges). |
| `--bidirectional`    | flag | `False`         | If true, use bidirectional GNN variant.                  |
| `--wandb`            | flag | `False`         | Enable logging to Weights & Biases.                     |
| `--fine_tuning`      | flag | `False`         | Use a pretrained autoencoder.                     |
| `--GNN`              | str  | `"SAGE"`        | GNN options: `SAGE` or `GIN`.                          |
| `--num_layers`       | int  | `1`             | Number of GNN layers.                                   |
| `--log_file_name`    | str  | `"default.txt"` | File to log results.                                    |
| `--pretrained_model` | str  | `"default.txt"` | Path to pretrained model (if fine-tuning).              |
| `--seed`             | int  | `1`             | Random seed for reproducibility.                        |

---

## Output

* Prints **F1 score** and **ROC-AUC** for train, validation, and test sets per fold.
* Logs results in the specified `--log_file_name` with columns:
* If `--wandb` is enabled, results are also logged to your wandb dashboard.

---
