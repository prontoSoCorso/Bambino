import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from momentfm import MOMENTPipeline
import config_ad as cfg

# Simple Masking Fallback
try:
    from momentfm.utils.masking import Masking
except ImportError:
    class Masking:
        def __init__(self, mask_ratio):
            self.mask_ratio = mask_ratio
        def generate_mask(self, x, input_mask):
            B, C, T = x.shape
            mask = torch.rand(B, 1, T) < self.mask_ratio
            return mask.long()

class EarlyStopper:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
            print("<-- New Best Validation Loss Achieved -->")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class MomentAnomalyDetector:
    def __init__(self):
        print(f"Loading MOMENT (Reconstruction) on {cfg.DEVICE}...")
        self.model = MOMENTPipeline.from_pretrained(
            cfg.MOMENT_MODEL,
            model_kwargs={
                'task_name': 'reconstruction',
                'n_channels': cfg.N_CHANNELS,
                'num_class': cfg.NUM_CLASSES
            }
        )
        self.model.init()
        self.model.to(cfg.DEVICE)
        
    def fine_tune(self, train_loader, val_loader):
        """
        Fine-tuning with Validation, Scheduler, and Early Stopping.
        """
        self.model.train()
        
        # Optimizer & Scheduler
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.LEARNING_RATE)
        # Reduce LR if validation loss plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=cfg.SCH_FACTOR, patience=cfg.PATIENCE//3)
        
        criterion = nn.MSELoss()
        mask_generator = Masking(mask_ratio=cfg.MASK_RATIO)
        early_stopper = EarlyStopper(patience=cfg.PATIENCE, min_delta=0.0001)
        
        print("Starting Fine-tuning...")
        
        for epoch in range(cfg.EPOCHS):
            # --- Training ---
            self.model.train()
            train_loss = 0
            for batch_x, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Train]"):
                batch_x = batch_x.to(cfg.DEVICE)
                input_mask = torch.ones(batch_x.shape[0], batch_x.shape[2]).to(cfg.DEVICE)
                
                n_batch, n_chan, n_time = batch_x.shape
                x_reshaped = batch_x.reshape(-1, 1, n_time)
                mask_reshaped = input_mask.repeat_interleave(n_chan, dim=0)
                
                mask = mask_generator.generate_mask(x=x_reshaped, input_mask=mask_reshaped).to(cfg.DEVICE).long()
                
                output = self.model(x_enc=x_reshaped, input_mask=mask_reshaped, mask=mask)
                loss = criterion(output.reconstruction, x_reshaped)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # --- Validation ---
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, _, _ in val_loader:
                    batch_x = batch_x.to(cfg.DEVICE)
                    
                    n_batch, n_chan, n_time = batch_x.shape
                    input_mask = torch.ones(n_batch, n_time).to(cfg.DEVICE)
                    
                    x_reshaped = batch_x.reshape(-1, 1, n_time)
                    mask_reshaped = input_mask.repeat_interleave(n_chan, dim=0)
                    
                    # Apply Masking in Validation too for consistent loss tracking
                    mask = mask_generator.generate_mask(x=x_reshaped, input_mask=mask_reshaped).to(cfg.DEVICE).long()
                    
                    output = self.model(x_enc=x_reshaped, input_mask=mask_reshaped, mask=mask)
                    loss = criterion(output.reconstruction, x_reshaped)
                    val_loss += loss.item()
                    
            avg_val_loss = val_loss / len(val_loader)
            
            # --- Updates ---
            print(f"Epoch {epoch+1} -> LR: {optimizer.param_groups[0]['lr']:.6f} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            
            scheduler.step(avg_val_loss)
            early_stopper(avg_val_loss)
            
            if early_stopper.early_stop:
                print("Early stopping triggered!")
                break
        
        print("Fine-tuning Complete.")

    def predict(self, loader):
        self.model.eval()
        mse_scores = []
        labels_list = []
        
        # separate lists to ensure we capture both classes
        vis_normal = []
        vis_anom = []
        MAX_VIS = 30 # Collect n examples of each class
        
        criterion = nn.MSELoss(reduction='none')
        
        print("Running Inference...")
        with torch.no_grad():
            for batch_x, batch_y, _ in tqdm(loader):
                batch_x = batch_x.to(cfg.DEVICE)
                batch_y_np = batch_y.cpu().numpy() # [B]
                
                # ... (Standard MOMENT Forward Pass) ...
                input_mask = torch.ones(batch_x.shape[0], batch_x.shape[2]).to(cfg.DEVICE)
                n_batch, n_chan, n_time = batch_x.shape
                x_reshaped = batch_x.reshape(-1, 1, n_time)
                mask_reshaped = input_mask.repeat_interleave(n_chan, dim=0)
                
                output = self.model(x_enc=x_reshaped, input_mask=mask_reshaped)
                pred = output.reconstruction # [B*C, 1, T]
                
                # Calculate Loss
                loss = criterion(pred, x_reshaped)
                loss = loss.mean(dim=2).reshape(n_batch, n_chan).mean(dim=1)
                
                mse_scores.extend(loss.cpu().numpy())
                labels_list.extend(batch_y_np)
                
                # --- NEW SAVING LOGIC ---
                # Reshape pred back to [B, C, T] for storage
                pred_full = pred.reshape(n_batch, n_chan, n_time).cpu().numpy()
                true_full = batch_x.cpu().numpy()
                
                # Iterate through batch and selectively save
                for k in range(n_batch):
                    lbl = batch_y_np[k]
                    
                    # If Normal (Class 1) and we need more
                    if lbl == cfg.NORMAL_CLASS and len(vis_normal) < MAX_VIS:
                        vis_normal.append({
                            'true': true_full[k], 
                            'pred': pred_full[k], 
                            'label': lbl
                        })
                        
                    # If Anomaly (Class 0) and we need more
                    elif lbl == cfg.ANOMALY_CLASS and len(vis_anom) < MAX_VIS:
                        vis_anom.append({
                            'true': true_full[k], 
                            'pred': pred_full[k], 
                            'label': lbl
                        })
        
        # Combine lists into the final dictionary format expected by vis_utils
        combined = vis_normal + vis_anom
        vis_data = {
            'true': [x['true'] for x in combined],
            'pred': [x['pred'] for x in combined],
            'label': [x['label'] for x in combined]
        }
                        
        return np.array(mse_scores), np.array(labels_list), vis_data