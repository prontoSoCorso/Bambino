import torch
import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm, trange
from collections import defaultdict
from sklearn.metrics import balanced_accuracy_score
import time, os, sys
import numpy as np
import logging
import optuna

# Aggiungo il percorso del progetto (Bambino) al sys.path
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "Bambino":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from _utils_ import models_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────────────────────────
# Training single epoch
# ────────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scaler, epoch, base_threshold):
    model.train()
    total_loss = 0.
    total_samples = 0
    correct_predictions = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch:03d}', leave=False)
    for batch_idx, (X_dict, y, _) in enumerate(pbar):
        # Move data to device
        X_dict = {k: v.to(device, non_blocking=True) for k, v in X_dict.items()}
        y = y.view(-1).to(device, dtype=torch.float, non_blocking=True)

        batch_size = y.size(0)

        optimizer.zero_grad(set_to_none=True)  # More efficient than not using "set_to_none"
        with autocast(device_type=str(device).split(':')[0]):
            logits = model(X_dict)
            loss = criterion(logits, y)

        # Training accuracy
        predictions = (torch.sigmoid(logits) > 0.75).float() # Applica sigmoide e soglia a 0.5 per ottenere predizioni binarie
        correct_predictions += (predictions == y).sum().item() # Confronta con il ground truth

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # Gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # Statistics
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Update progress bar
        current_train_acc = correct_predictions / total_samples if total_samples > 0 else 0
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg': f'{total_loss/total_samples:.4f}',
            'GradNorm': f'{grad_norm:.3f}',
            'TrainAcc': f'{current_train_acc:.4f}'
        })

    return total_loss / total_samples, correct_predictions / total_samples


# ────────────────────────────────────────────────────────────────────────────────
# General Training
# ────────────────────────────────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, config, model_save_path, trial=None):
    """
    Train the LSTM-FCN model with optional Optuna trial support
    
    Args:
        model: The LSTM-FCN model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration object with hyperparameters
        model_save_path: Path to save the best model
        trial: Optuna trial object (None for normal training)
    
    Returns:
        model: Trained model
        history: Training history
    """
    # Determine if we're in optimization mode
    optimization_mode = trial is not None

    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=getattr(config, 'lstmfcn_learning_rate', 1e-5),
        weight_decay=1e-4
        )
    
    # Setup scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=getattr(config, 'lstmfcn_max_lr', 1e-3),
        steps_per_epoch=len(train_loader),
        epochs=getattr(config, 'lstmfcn_num_epochs', 100),
        pct_start=getattr(config, 'lstmfcn_pct_start', 0.3),
        anneal_strategy=getattr(config, 'lstmfcn_anneal_strategy', 'cos')
        )
    
    # Setup loss function
    use_focal = getattr(config, 'lstmfcn_use_focal', True)
    if use_focal:
        criterion = models_utils.FocalLoss(alpha=1, gamma=2)
    else:
        criterion = nn.BCEWithLogitsLoss()

    scaler = GradScaler('cuda' if 'cuda' in str(device) else 'cpu')

    lstmfcn_num_epochs = getattr(config, 'lstmfcn_num_epochs', 100)
    if not optimization_mode:
        print(f"\n🚀 Starting training for {lstmfcn_num_epochs} epochs...")
        print(f"📊 Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

    # Training tracking
    best_metric = float('inf')
    epochs_no_improve = 0
    train_history = []
    ref_metric = 'brier'
    base_threshold = 0.5

    for epoch in trange(lstmfcn_num_epochs, desc="Training Progress", disable=optimization_mode):
        epoch_start = time.time()
        
        # Training phase
        train_loss, train_acc = train_one_epoch(model=model, 
                                                loader=train_loader,
                                                optimizer=optimizer,
                                                criterion=criterion,
                                                scaler=scaler, 
                                                epoch=epoch, 
                                                base_threshold=base_threshold)
        # validation
        y_true, y_prob, val_loss = evaluate_model(model, val_loader, desc=f"Validation Epoch {epoch:03d}")
        y_pred = (np.array(y_prob) >= base_threshold).astype(int)
        val_metrics = models_utils.calculate_metrics(y_true, y_pred, y_prob)
        
        # Scheduler step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Logging and tracking
        epoch_time = time.time() - epoch_start
        
        # Choose primary metric for model selection
        primary_metric = val_metrics.get(ref_metric, val_loss)
        
        # Store metrics in history (as dictionaries for compatibility)
        epoch_data = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_mcc': val_metrics.get('MCC', 0),
            'val_brier': val_metrics.get('brier', 0),
            'val_balanced_acc': val_metrics.get('balanced_accuracy', 0),
            'val_f1': val_metrics.get('f1_macro', 0),
            'val_f1_macro': val_metrics.get('f1_macro', 0),
            'val_f1_weighted': val_metrics.get('f1_weighted', 0),
            'learning_rate': current_lr,
            'epoch_time': epoch_time
        }
        train_history.append(epoch_data)

        # Detailed logging
        logging.info(
            f"Epoch {epoch:03d}/{lstmfcn_num_epochs-1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"mcc: {val_metrics.get('MCC', 0):.4f} | "
            f"Brier: {val_metrics.get('brier', 0):.4f} | "
            f"Bal Acc: {val_metrics.get('balanced_accuracy', 0):.4f} | "
            f"f1_macro: {val_metrics.get('f1_macro', 0):.4f} | "
            f"f1_weighted: {val_metrics.get('f1_weighted', 0):.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Model checkpointing
        if primary_metric + 1e-4 < best_metric:
            best_metric = primary_metric
            epochs_no_improve = 0

            # Save best model (only if not in optimization mode)
            if not optimization_mode:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_metric': best_metric,
                    'train_history': train_history,
                    'params': dict(
                        batch_size  =config.lstmfcn_batch_size,
                        lr          =config.lstmfcn_learning_rate,
                        epochs      =lstmfcn_num_epochs,
                        filter_sizes=config.cnn_filter_sizes,
                        kernel_sizes=config.cnn_kernel_sizes,
                        lstm_hidden =config.lstm_hidden_dim
                    )
                }, model_save_path)
                logging.info(f"New Best Model saved with val_{ref_metric}={primary_metric:.4f}")
            else:
                # For optimization mode, save to the trial-specific path
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_metric': best_metric,
                    'train_history': train_history,
                }, model_save_path)

        else:
            epochs_no_improve += 1
        
        # Handle Optuna pruning
        if optimization_mode and trial is not None:
            # Report intermediate value for pruning
            trial.report(val_metrics.get('balanced_accuracy', 0), epoch)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Early stopping
        if epochs_no_improve >= config.lstmfcn_early_stopping_patience:
            logging.info(f"🛑 Early stopping triggered after {epochs_no_improve} epochs without improvement")
            break

    if not optimization_mode and os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"📥 Loaded best model from epoch {checkpoint['epoch']} with metric {checkpoint['best_metric']:.4f}")
    
    return model, train_history


# ────────────────────────────────────────────────────────────────────────────────
# Model evaluation
# ────────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, loader, threshold=0.75, desc="Evaluating"):
    model.eval()
    y_true, y_prob = [], []
    total_loss = 0.0
    total_samples = 0
    
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        pbar = tqdm(loader, desc=desc, leave=False)
        for X_dict, y, _ in pbar:
            X_dict = {k: v.to(device, non_blocking=True) for k, v in X_dict.items()}
            y_batch = y.view(-1).to(device, dtype=torch.float, non_blocking=True)
            
            with autocast(device_type=str(device).split(':')[0]):
                logits = model(X_dict)
                loss = criterion(logits, y_batch)
            
            probs = torch.sigmoid(logits).cpu().numpy()
            y_true.extend(y.view(-1).cpu().numpy())
            y_prob.extend(probs)
            
            total_loss += loss.item() * y_batch.size(0)
            total_samples += y_batch.size(0)
            
            pbar.set_postfix({'Loss': f'{total_loss/total_samples:.4f}'})
    
    return y_true, y_prob, total_loss / total_samples


# ────────────────────────────────────────────────────────────────────────────────
# To find the best threshold at the end of the training
# ────────────────────────────────────────────────────────────────────────────────
def find_best_threshold(model, val_loader, thresholds=np.linspace(0,1,101)):
    """
    Evaluate the model on `val_loader` across multiple candidate thresholds
    to maximize balanced accuracy. Returns the best threshold.
    """
    y_true, y_prob, _ = evaluate_model(model, val_loader, desc="Best Threshold Search")
    best_thr, best_bal = 0.75, 0.
    for thr in thresholds:
        y_pred = (np.array(y_prob) >= thr).astype(int)
        bal = balanced_accuracy_score(y_true, y_pred)
        if bal > best_bal:
            best_bal, best_thr = bal, thr
    logging.info(f"Best threshold on val: {best_thr:.2f} (bal_acc={best_bal:.3f})")
    return best_thr
