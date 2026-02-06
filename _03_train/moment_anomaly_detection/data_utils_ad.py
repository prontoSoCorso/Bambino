import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config_ad as cfg

# --- Path Setup ---
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

from config import settings as global_settings
from DataUtils.BoaOpenFaceDataset import BoaOpenFaceDataset

class ResizedDataset(Dataset):
    def __init__(self, data_list, labels, meta_list):
        self.data = data_list 
        self.labels = labels
        self.meta = meta_list
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y, self.meta[idx]

def resize_signal(X_raw, target_len=512):
    """Resizes [T, C] -> [C, 512]."""
    X_t = X_raw.T 
    C, T_orig = X_t.shape
    X_new = np.zeros((C, target_len), dtype=np.float32)
    orig_idx = np.linspace(0, T_orig-1, T_orig)
    targ_idx = np.linspace(0, T_orig-1, target_len)
    for c in range(C):
        X_new[c, :] = np.interp(targ_idx, orig_idx, X_t[c, :])
    return X_new

def normalize_instance(X):
    """Z-score normalization per instance (Channel-wise)."""
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X.T).T 
    return X_norm

def load_from_boa_dataset(dataset_path):
    print(f"   -> Loading: {os.path.basename(dataset_path)}...")
    try:
        ds = BoaOpenFaceDataset.load_dataset(dataset_path)
    except Exception as e:
        print(f"      [Error] Could not load dataset: {e}")
        return [], [], []

    X_list = []
    y_list = []
    meta_list = []

    for i, inst in enumerate(ds.instances):
        try:
            g = inst.gaze_info 
            h = inst.head_info 
            f = inst.face_info 
            combined = np.hstack([g, h, f]) # [T, 38]
            
            resized = resize_signal(combined, target_len=cfg.TARGET_LEN)
            normed = normalize_instance(resized)
            
            label = int(inst.trial_type)
            meta = {
                'pt_id': inst.pt_id, 
                'age': inst.age, 
                'sex': inst.sex, 
                'trial_id': inst.trial_id, 
                'audio': inst.audio
            }
            
            X_list.append(normed)
            y_list.append(label)
            meta_list.append(meta)
            
        except Exception as e:
            continue
            
    return X_list, y_list, meta_list

def get_ad_dataloaders(test_subjects_count=5):
    """
    Creates Subject-Independent splits for Anomaly Detection.
    
    Args:
        test_subjects_count (int): Number of babies to hold out completely for testing.
    """
    # 1. Load Everything
    ds_type = 'normalized' 
    train_path = global_settings.get_dataset_path(ds_type, global_settings.training_filename)
    val_path   = global_settings.get_dataset_path(ds_type, global_settings.validation_filename)
    test_path  = global_settings.get_dataset_path(ds_type, global_settings.test_filename)
    
    print("Loading and Pooling all datasets...")
    X_all, y_all, meta_all = [], [], []
    for path in [train_path, val_path, test_path]:
        x, y, m = load_from_boa_dataset(path)
        X_all.extend(x)
        y_all.extend(y)
        meta_all.extend(m)
        
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    
    # 2. Extract Unique Subjects
    all_subjects = np.unique([m['pt_id'] for m in meta_all])
    print(f"Total Unique Subjects found: {len(all_subjects)}")
    
    # 3. Split Subjects
    # Step A: Hold out N subjects for TEST
    train_val_subjs, test_subjs = train_test_split(
        all_subjects, 
        test_size=test_subjects_count, 
        random_state=cfg.SEED, 
        shuffle=True
    )
    
    # Step B: Hold out 10% of remaining subjects for VALIDATION
    # (Using subjects for validation ensures Early Stopping works on generalization, not memorization)
    train_subjs, val_subjs = train_test_split(
        train_val_subjs,
        test_size=0.10, # 10% of the training pool
        random_state=cfg.SEED,
        shuffle=True
    )
    
    print(f"Subject Split:")
    print(f"  Train Subjects: {len(train_subjs)}")
    print(f"  Val Subjects:   {len(val_subjs)}")
    print(f"  Test Subjects:  {len(test_subjs)}")

    # 4. Filter Data based on Subjects & Class
    
    # Helper to get indices
    def get_indices(subjects, keep_only_normal=True):
        indices = []
        for i, meta in enumerate(meta_all):
            if meta['pt_id'] in subjects:
                # If training/val, we ONLY want Normal class
                if keep_only_normal:
                    if y_all[i] == cfg.NORMAL_CLASS:
                        indices.append(i)
                # If test, we want EVERYTHING (Normal + Anomaly)
                else:
                    indices.append(i)
        return np.array(indices)

    # Train Set: Train Subjects (Normal Only)
    train_idx = get_indices(train_subjs, keep_only_normal=True)
    
    # Val Set: Val Subjects (Normal Only)
    val_idx = get_indices(val_subjs, keep_only_normal=True)
    
    # Test Set: Test Subjects (All Data)
    test_idx = get_indices(test_subjs, keep_only_normal=False)
    
    # 5. Build Datasets
    train_ds = ResizedDataset(X_all[train_idx], y_all[train_idx], [meta_all[i] for i in train_idx])
    val_ds   = ResizedDataset(X_all[val_idx],   y_all[val_idx],   [meta_all[i] for i in val_idx])
    test_ds  = ResizedDataset(X_all[test_idx],  y_all[test_idx],  [meta_all[i] for i in test_idx])
    
    # 6. Loaders
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    print(f"\nSubject-Independent AD Data Specs:")
    print(f"  Train (Normal Only): {len(train_ds)} samples")
    print(f"  Val   (Normal Only): {len(val_ds)} samples")
    print(f"  Test  (Mixed):       {len(test_ds)} samples")
    # transform test_ds.labels to numpy array for counting
    print(f"      -> Normals:   {np.sum(np.array(test_ds.labels) == cfg.NORMAL_CLASS)}")
    print(f"      -> Anomalies: {np.sum(np.array(test_ds.labels) == cfg.ANOMALY_CLASS)}")
    
    return train_loader, val_loader, test_loader