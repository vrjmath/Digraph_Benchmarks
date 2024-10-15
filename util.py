import torch

def f1_score(y_true, y_pred, threshold=0.5):
 
    y_pred_binary = (y_pred > threshold).float()
    
    TP = (y_true * y_pred_binary).sum().item()
    FP = ((1 - y_true) * y_pred_binary).sum().item()
    FN = (y_true * (1 - y_pred_binary)).sum().item()
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1