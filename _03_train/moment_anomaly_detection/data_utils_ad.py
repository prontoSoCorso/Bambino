import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import config_ad as cfg

# --- Path Setup to allow importing DataUtils ---
def find_project_root(name='Bambino', start=None):
    if start is None: start = os.getcwd()
    parent = start
    while True:
        if os.path.basename(parent) == name: return parent
        if os.path.dirname(parent) == parent: return None
        parent = os.path.dirname(parent)

PROJECT_ROOT = find_project_root()
if PROJECT_ROOT: 
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)

# Now we can import your project modules
from config import settings as global_settings
from DataUtils.BoaOpenFaceDataset import BoaOpenFaceDataset

class ResizedDataset(Dataset):
    """
    Simple wrapper for the resized/pooled data for MOMENT AD.
    """
    def __init__(self, data_list, labels, meta_list):
        self.data = data_list # List of [38, 512] arrays
        self.labels = labels
        self.meta = meta_list
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return FloatTensor for model, Label for evaluation
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y, self.meta[idx]

def resize_signal(X_raw, target_len=512):
    """
    Resizes [T, C] -> [C, 512] using linear interpolation.
    MOMENT expects the channel dimension first.
    """
    # X_raw is [Time, Channels] -> Transpose to [Channels, Time]
    X_t = X_raw.T 
    C, T_orig = X_t.shape
    
    X_new = np.zeros((C, target_len), dtype=np.float32)
    orig_idx = np.linspace(0, T_orig-1, T_orig)
    targ_idx = np.linspace(0, T_orig-1, target_len)
    
    for c in range(C):
        X_new[c, :] = np.interp(targ_idx, orig_idx, X_t[c, :])
        
    return X_new

def load_from_boa_dataset(dataset_path):
    """
    Loads a .pkl file using BoaOpenFaceDataset and extracts instances.
    Returns: lists of X (resized), y, meta
    """
    print(f"   -> Loading: {os.path.basename(dataset_path)}...")
    try:
        # Use the static load method from your class
        ds = BoaOpenFaceDataset.load_dataset(dataset_path)
    except Exception as e:
        print(f"      [Error] Could not load dataset: {e}")
        return [], [], []

    X_list = []
    y_list = []
    meta_list = []

    # Iterate directly over the instances list in the loaded object
    for i, inst in enumerate(ds.instances):
        try:
            # 1. Access properties directly from OpenFaceInstance
            # Note: These are already numpy arrays handled by OpenFaceInstance.__init__
            g = inst.gaze_info # [T, 8]
            h = inst.head_info # [T, 13]
            f = inst.face_info # [T, 17]
            
            # 2. Stack Modalities -> [T, 38]
            combined = np.hstack([g, h, f])
            
            # 3. Resize to MOMENT format [38, 512]
            resized = resize_signal(combined, target_len=cfg.TARGET_LEN)
            
            # 4. Extract Label and Metadata
            label = int(inst.trial_type) # 0=Control, 1=Stimulus
            
            meta = {
                'pt_id': inst.pt_id,
                'age': inst.age,
                'sex': inst.sex,
                'trial_id': inst.trial_id,
                'audio': inst.audio
            }
            
            X_list.append(resized)
            y_list.append(label)
            meta_list.append(meta)
            
        except Exception as e:
            print(f"      Skipping instance {i}: {e}")
            continue
            
    return X_list, y_list, meta_list

def get_ad_dataloaders():
    """
    Pools Train + Val + Test (custom split) and creates:
    Train: 75% of Normal (Stimulus/Class 1)
    Test: 25% of Normal + 100% of Anomalies (Control/Class 0)
    """
    # 1. Get paths using your global_settings
    # We use 'normalized' or 'augmented_normalized' as defined in your config_moment/config_ad
    # Defaulting to 'normalized' for AD usually as we want clean reconstruction targets
    ds_type = 'normalized' 
    
    train_path = global_settings.get_dataset_path(ds_type, global_settings.training_filename)
    val_path   = global_settings.get_dataset_path(ds_type, global_settings.validation_filename)
    test_path  = global_settings.get_dataset_path(ds_type, global_settings.test_filename)
    
    # 2. Load and Pool everything
    print("Loading all datasets...")
    X_all, y_all, meta_all = [], [], []
    
    for path in [train_path, val_path, test_path]:
        x, y, m = load_from_boa_dataset(path)
        X_all.extend(x)
        y_all.extend(y)
        meta_all.extend(m)
        
    X_all = np.array(X_all) # [N, 38, 512]
    y_all = np.array(y_all) # [N]
    # Keep meta_all as list
    
    # 3. Separate Indices
    idx_normal = np.where(y_all == cfg.NORMAL_CLASS)[0]
    idx_anom   = np.where(y_all == cfg.ANOMALY_CLASS)[0]
    
    print(f"Total Samples: {len(X_all)}")
    print(f"  - Normal (Class {cfg.NORMAL_CLASS}): {len(idx_normal)}")
    print(f"  - Anomaly (Class {cfg.ANOMALY_CLASS}): {len(idx_anom)}")
    
    # 4. Split Normal into Train/Test
    # We shuffle the normals to ensure the train set is representative
    train_idx, test_normal_idx = train_test_split(
        idx_normal, 
        train_size=cfg.TRAIN_SPLIT_RATIO, 
        random_state=cfg.SEED,
        shuffle=True
    )
    
    # 5. Construct Final Sets
    # Train: Only Normal
    train_X = X_all[train_idx]
    train_y = y_all[train_idx]
    train_meta = [meta_all[i] for i in train_idx]
    
    # Test: Remaining Normal + All Anomalies
    test_idx_all = np.concatenate([test_normal_idx, idx_anom])
    test_X = X_all[test_idx_all]
    test_y = y_all[test_idx_all]
    test_meta = [meta_all[i] for i in test_idx_all]
    
    # 6. Create Datasets & Loaders
    train_ds = ResizedDataset(train_X, train_y, train_meta)
    test_ds  = ResizedDataset(test_X, test_y, test_meta)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    print(f"\nAD Data Preparation Complete:")
    print(f"  -> Train Loader: {len(train_ds)} samples (Clean Normals)")
    print(f"  -> Test Loader:  {len(test_ds)} samples ({len(test_normal_idx)} Normals, {len(idx_anom)} Anomalies)")
    
    return train_loader, test_loader