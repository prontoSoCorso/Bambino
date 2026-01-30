import torch
import numpy as np
from tqdm import tqdm
from momentfm import MOMENTPipeline
import config_moment as cfg
import pandas as pd

def get_moment_model(task='embedding'):
    """Loads MOMENT model ensuring correct device placement."""
    print(f"Loading MOMENT ({task}) on {cfg.device}...")
    model = MOMENTPipeline.from_pretrained(
        cfg.moment_model_name,
        model_kwargs={
            'task_name': task,
            'n_channels': cfg.n_channels,
            'num_class': cfg.num_classes
        }
    )
    model.init()
    model.to(cfg.device)
    model.eval()
    return model

class MomentFeatureExtractor:
    def __init__(self):
        self.model = get_moment_model(task='embedding')
        self.target_len = cfg.target_len

    def resize_data(self, X_list):
        """
        Resizes a list of arrays [T, C] to [N, C, 512] using interpolation.
        MOMENT expects fixed length 512.
        """
        resized_list = []
        for X in X_list:
            # X is [T, C] -> Transpose to [C, T]
            X_t = X.T 
            C, T_orig = X_t.shape
            
            # Interpolate to target_len (512)
            X_new = np.zeros((C, self.target_len), dtype=np.float32)
            orig_idx = np.linspace(0, T_orig-1, T_orig)
            targ_idx = np.linspace(0, T_orig-1, self.target_len)
            
            for c in range(C):
                X_new[c, :] = np.interp(targ_idx, orig_idx, X_t[c, :])
            
            resized_list.append(X_new)
        
        return np.stack(resized_list, axis=0) # [N, C, 512]

    def extract(self, dataset):
        """
        Iterates over dataset, resizes, and runs MOMENT inference.
        Returns: embeddings [N, 1024], metadata_df
        """
        # 1. Collect Data
        X_raw = []
        meta_data = []
        labels = []
        
        print("Collecting and resizing data...")
        for i in range(len(dataset)):
            try:
                # Handle potential getitem variations
                if hasattr(dataset, 'instances'):
                    inst = dataset.instances[i]
                    # Stack modalities [Gaze, Head, Face] -> [T, 38]
                    # Ensure they are numpy arrays
                    g = np.array(inst.gaze_info)
                    h = np.array(inst.head_info)
                    f = np.array(inst.face_info)
                    combined = np.hstack([g, h, f])
                    
                    meta_data.append({'pt_id': inst.pt_id, 'age': inst.age, 'sex': inst.sex})
                    labels.append(int(inst.trial_type))
                else:
                    # Fallback for simple TensorDatasets
                    data, y, _ = dataset[i]
                    combined = data.numpy() # [T, C]
                    meta_data.append({'pt_id': f"sample_{i}", 'age': 0, 'sex': 0})
                    labels.append(int(y))
                
                X_raw.append(combined)
            except Exception as e:
                print(f"Skipping index {i}: {e}")
                continue

        # 2. Resize
        X_tensor = torch.tensor(self.resize_data(X_raw), dtype=torch.float32)
        
        # 3. Inference Loop
        embeddings = []
        print(f"Extracting Embeddings (Batch Size {cfg.batch_size})...")
        
        with torch.no_grad():
            for i in tqdm(range(0, len(X_tensor), cfg.batch_size)):
                batch = X_tensor[i : i + cfg.batch_size].to(cfg.device)
                
                # Input Mask (1 = observe)
                mask = torch.ones(batch.shape[0], batch.shape[2]).to(cfg.device)
                
                # Forward
                output = self.model(x_enc=batch, input_mask=mask)
                
                # Pool: MOMENT returns [Batch, Time, Dim]. We mean-pool over time.
                emb = output.embeddings.detach().cpu().numpy()
                if emb.ndim == 3:
                    mean_emb = emb.mean(axis=1)
                    std_emb = emb.std(axis=1)
                    emb = np.hstack([mean_emb, std_emb])
                
                embeddings.append(emb)
        
        return np.concatenate(embeddings, axis=0), np.array(labels), pd.DataFrame(meta_data)