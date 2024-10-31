import pandas as pd
import numpy as np
import scipy.stats as stats

def calculate_confidence_interval(column_data):
    n = len(column_data) 
    mean = np.mean(column_data) 
    std_dev = np.std(column_data, ddof=1) 

    z_value = stats.norm.ppf(0.975)
    margin_of_error = z_value * (std_dev / np.sqrt(n))

    return std_dev

columns = ["Num Threads", "Batch Size", "Epochs", "Patience", "Undirected", "Bidirectional", "GNN", "# Layers", "seed", "F1 Val", "ROC Val", "F1 Test", "ROC Test", "ts"]

#df = pd.read_csv('final_logs/downstream_log_unSAGE3.txt', delim_whitespace=True, header=None, names=columns, skiprows=1)
df = pd.read_csv('finetune_logs/freeze_finetune_directed_prediction_SAGE1k.txt', delim_whitespace=True, header=None, names=columns, skiprows=1)
#df = pd.read_csv('downstream_logs/SAGE/downstream_log.txt', delim_whitespace=True, header=None, names=columns, skiprows=1)
#df = pd.read_csv('downstream_logs/downstream_log_SAGEC.txt', delim_whitespace=True, header=None, names=columns, skiprows=1)

print(f"Num Samples:  {len(df['F1 Val'])}")
print(f"Val F1:       {df['F1 Val'].mean():.3f} ± {calculate_confidence_interval(df['F1 Val']):.3f}")
print(f"Val ROC-AUC:  {df['ROC Val'].mean():.3f} ± {calculate_confidence_interval(df['ROC Val']):.3f}")
print(f"Test F1:      {df['F1 Test'].mean():.3f} ± {calculate_confidence_interval(df['F1 Test']):.3f}")
print(f"Test ROC-AUC: {df['ROC Test'].mean():.3f} ± {calculate_confidence_interval(df['ROC Test']):.3f}")

#df.to_csv('output_file.csv', index=False)
