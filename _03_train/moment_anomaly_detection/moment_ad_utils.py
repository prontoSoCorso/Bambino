import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from momentfm import MOMENTPipeline
# Improvised Masking if not importable, though it should be in momentfm.utils.masking
try:
    from momentfm.utils.masking import Masking
except ImportError:
    # Fallback simple masking class if specific import fails
    class Masking:
        def __init__(self, mask_ratio):
            self.mask_ratio = mask_ratio
        def generate_mask(self, x, input_mask):
            # x: [B, C, T]
            B, C, T = x.shape
            mask = torch.rand(B, 1, T) < self.mask_ratio # Simple random mask
            return mask.long()

import config_ad as cfg

class MomentAnomalyDetector:
    def __init__(self):
        print(f"Loading MOMENT (Reconstruction) on {cfg.DEVICE}...")
        self.model = MOMENTPipeline.from_pretrained(
            cfg.MOMENT_MODEL,
            model_kwargs={
                'task_name': 'reconstruction',
                'n_channels': cfg.N_CHANNELS,
                'num_class': 2 # Not used for recon, but required arg sometimes
            }
        )
        self.model.init()
        self.model.to(cfg.DEVICE)
        
    def fine_tune(self, train_loader):
        """
        Fine-tunes the model on the normal data using Masked Mean Squared Error.
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.LEARNING_RATE)
        criterion = nn.MSELoss()
        mask_generator = Masking(mask_ratio=cfg.MASK_RATIO)
        
        print("Starting Fine-tuning...")
        for epoch in range(cfg.EPOCHS):
            total_loss = 0
            for batch_x, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}"):
                batch_x = batch_x.to(cfg.DEVICE) # [B, C, 512]
                
                # Input mask (All 1s usually, unless padding is needed)
                input_mask = torch.ones(batch_x.shape[0], batch_x.shape[2]).to(cfg.DEVICE)
                
                # MOMENT expects [B, 1, 512] for univariate masking logic or similar
                # However, for multivariate, we usually pass standard shape.
                # The tutorial reshapes to (Batch*n_channels, 1, T). Let's follow that pattern
                # to ensure the PatchMasking works as expected in their library.
                
                n_batch, n_chan, n_time = batch_x.shape
                
                # Reshape for masking logic: Treat every channel as an independent time series 
                # (Standard practice for MOMENT pre-training)
                x_reshaped = batch_x.reshape(-1, 1, n_time) # [B*C, 1, T]
                mask_reshaped = input_mask.repeat_interleave(n_chan, dim=0) # [B*C, T]
                
                # Generate Mask
                mask = mask_generator.generate_mask(
                    x=x_reshaped, input_mask=mask_reshaped
                ).to(cfg.DEVICE).long()
                
                # Forward Pass
                output = self.model(x_enc=x_reshaped, input_mask=mask_reshaped, mask=mask)
                
                # Reconstruction Loss
                # We compare output.reconstruction with x_reshaped
                loss = criterion(output.reconstruction, x_reshaped)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.6f}")

    def predict(self, loader):
        """
        Runs inference (No Masking) and calculates MSE per sample.
        Returns: (MSE_scores, Labels, Dictionary of {Original, Reconstructed})
        """
        self.model.eval()
        mse_scores = []
        labels_list = []
        
        # Store a few examples for visualization (First 5)
        vis_data = {'true': [], 'pred': [], 'label': []}
        
        criterion = nn.MSELoss(reduction='none') # Keep batch dim
        
        print("Running Inference...")
        with torch.no_grad():
            for batch_x, batch_y, _ in tqdm(loader):
                batch_x = batch_x.to(cfg.DEVICE)
                batch_y = batch_y.cpu().numpy()
                
                # No Masking during inference (reconstruct everything)
                input_mask = torch.ones(batch_x.shape[0], batch_x.shape[2]).to(cfg.DEVICE)
                
                # We still reshape to match the training behavior/model expectation
                n_batch, n_chan, n_time = batch_x.shape
                x_reshaped = batch_x.reshape(-1, 1, n_time)
                mask_reshaped = input_mask.repeat_interleave(n_chan, dim=0)
                
                output = self.model(x_enc=x_reshaped, input_mask=mask_reshaped)
                
                # Output is [B*C, 1, T]
                pred = output.reconstruction
                
                # Calculate Loss per channel/time
                loss = criterion(pred, x_reshaped) # [B*C, 1, T]
                
                # Aggregation Strategy:
                # 1. Mean over Time
                # 2. Mean over Channels (Reshape back to B, C)
                loss = loss.mean(dim=2) # [B*C, 1] (Mean MSE per channel)
                loss = loss.reshape(n_batch, n_chan) # [B, C]
                sample_mse = loss.mean(dim=1) # [B] (Average MSE over all channels)
                
                mse_scores.extend(sample_mse.cpu().numpy())
                labels_list.extend(batch_y)
                
                # Store visualization data (5 examples for positive class and 5 for negative class)
                for i in range(batch_x.shape[0]):
                    if len(vis_data['true']) < 10:
                        if (batch_y[i] == cfg.NORMAL_CLASS and sum(np.array(vis_data['label']) == cfg.NORMAL_CLASS) < 5) or \
                           (batch_y[i] == cfg.ANOMALY_CLASS and sum(np.array(vis_data['label']) == cfg.ANOMALY_CLASS) < 5):
                            # Reshape pred back to [B, C, T]
                            pred_reshaped = pred.reshape(n_batch, n_chan, n_time)
                            vis_data['true'].append(batch_x[i].cpu().numpy())
                            vis_data['pred'].append(pred_reshaped[i].squeeze(1).cpu().numpy())
                            vis_data['label'].append(batch_y[i])
                        
        return np.array(mse_scores), np.array(labels_list), vis_data