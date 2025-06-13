import os
import logging
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, 
                             cohen_kappa_score, brier_score_loss, 
                             confusion_matrix, f1_score,
                             roc_curve, auc, 
                             precision_score, recall_score, 
                             matthews_corrcoef)

import os
import torch
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────────────────
# Configure logging
# ────────────────────────────────────────────────────────────────────────────────
def config_logging(log_dir, log_filename):
    os.makedirs(log_dir, exist_ok=True)
    complete_filename = os.path.join(log_dir, log_filename)
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(complete_filename)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # Add handlers
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


# ────────────────────────────────────────────────────────────────────────────────
# Sampler 
# ────────────────────────────────────────────────────────────────────────────────
def get_balanced_sampler(dataset, num_classes):
    """Improved balanced sampling with class analysis"""
    labels = []
    for _, y, _ in tqdm(dataset, desc="Computing class weights", leave=False):
        labels.append(int(y.item()))
    
    labels = np.array(labels)
    class_counts = np.bincount(labels, minlength=num_classes)
    
    print(f"Class distribution: {dict(enumerate(class_counts))}")
    print(f"Class ratios: {class_counts / len(labels)}")
    
    # Inverse frequency weighting with smoothing
    total_samples = len(labels)
    class_weights = total_samples / (num_classes * class_counts + 1e-8)
    
    sample_weights = [class_weights[label] for label in labels]
    
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler

# ────────────────────────────────────────────────────────────────────────────────
# Focal Loss implementation
# ────────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
# ────────────────────────────────────────────────────────────────────────────────
# Model metrics
# ────────────────────────────────────────────────────────────────────────────────
def calculate_metrics(y_true, y_pred, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc,
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "brier": brier_score_loss(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, average='binary', zero_division=0),  # Binary F1 (positive class)
        "f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),  # Macro-average F1
        "f1_weighted": f1_score(y_true, y_pred, average='weighted', zero_division=0),  # Weighted-average F1
        "conf_matrix": confusion_matrix(y_true, y_pred),
        "fpr": fpr,
        "tpr": tpr
    }

    decimals = 4
    for key, value in metrics.items():
        if isinstance(value, float):
            metrics[key] = round(value, decimals)
        elif isinstance(value, np.ndarray): # Gestisce array NumPy
            metrics[key] = np.round(value, decimals) # Arrotonda gli elementi dell'array

    return metrics