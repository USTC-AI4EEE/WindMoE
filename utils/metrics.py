# utils/metrics.py

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score


class CustomLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(CustomLoss, self).__init__()
        self.loss_fn = nn.HuberLoss(delta=delta, reduction='mean')

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        
        return self.loss_fn(y_pred, y_true)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    y_pred = np.clip(y_pred, 0, None)
    
    metrics = {}
    metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    metrics['rmse'] = np.sqrt(np.mean((y_true - y_pred)**2))

    return metrics