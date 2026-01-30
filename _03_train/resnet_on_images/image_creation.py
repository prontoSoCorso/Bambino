import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# --- Project Setup ---
def find_project_root(name='Bambino', start=None):
    if start is None: start = os.getcwd()
    parent_dir = start
    while True:
        if os.path.basename(parent_dir) == name: return parent_dir
        new_parent = os.path.dirname(parent_dir)
        if new_parent == parent_dir: return None
        parent_dir = new_parent

PROJECT_ROOT = find_project_root('Bambino')
if PROJECT_ROOT: sys.path.append(PROJECT_ROOT)

from config import settings
from DataUtils.BoaOpenFaceDataset import BoaOpenFaceDataset

# --- Config ---
current_file_dir = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
DATASET_TYPE = 'normalized' # 'augmented_normalized' or 'normalized'
OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "_03_train", current_file_dir ,f"resnet_dataset_{DATASET_TYPE}")

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
IMAGE_SIZE = 224

def compute_gasf(X):
    """Compute Gramian Angular Summation Field."""
    # Clip to safety range [-1, 1]
    X_clipped = np.clip(X, -1, 1) 
    Sin_phi = np.sqrt(1 - X_clipped**2)
    # GASF Formula
    GASF = np.outer(X_clipped, X_clipped) - np.outer(Sin_phi, Sin_phi)
    return GASF

def process_split(dataset, split_name, pcas=None):
    save_dir = os.path.join(OUTPUT_BASE_DIR, "images", split_name)
    os.makedirs(save_dir, exist_ok=True)
    
    metadata_records = []
    
    # --- PCA FITTING (Train Only) ---
    if pcas is None:
        print(f"[{split_name}] Collecting data for Per-Modality PCA...")
        data_g, data_h, data_f = [], [], []
        
        # Collect samples (Limit to 2000 for speed)
        # We manually access .instances to avoid triggering __getitem__ crash
        limit = min(len(dataset), 2000)
        
        for i in range(limit):
            try:
                # Try standard access first
                X_dict, _, _ = dataset[i]
                data_g.append(X_dict['g'].numpy()) 
                data_h.append(X_dict['h'].numpy()) 
                data_f.append(X_dict['f'].numpy()) 
            except Exception:
                # Fallback: manual extraction if __getitem__ fails
                inst = dataset.instances[i]
                data_g.append(np.array(inst.gaze_info))
                data_h.append(np.array(inst.head_info))
                data_f.append(np.array(inst.face_info))

        # Fit 3 Separate PCAs
        print(f"[{split_name}] Fitting 3 separate PCAs...")
        
        pca_g = PCA(n_components=1).fit(np.concatenate(data_g, axis=0))
        pca_h = PCA(n_components=1).fit(np.concatenate(data_h, axis=0))
        pca_f = PCA(n_components=1).fit(np.concatenate(data_f, axis=0))
        
        pcas = {'g': pca_g, 'h': pca_h, 'f': pca_f}
        
        print("Explained Variance Ratios (PC1):")
        print(f"  Gaze: {pca_g.explained_variance_ratio_[0]:.4f}")
        print(f"  Head: {pca_h.explained_variance_ratio_[0]:.4f}")
        print(f"  Face: {pca_f.explained_variance_ratio_[0]:.4f}")

    # Scaler for GASF requirements [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    
    print(f"[{split_name}] Generating Semantic RGB Images...")
    
    for idx in tqdm(range(len(dataset))):
        try:
            X_dict, y, extras = dataset[idx]
            
            # Extract Metadata
            if hasattr(dataset, 'instances'):
                inst = dataset.instances[idx]
                pt_id = inst.pt_id
                age = float(inst.age) if inst.age is not None else np.nan
                sex = int(inst.sex) if hasattr(inst, 'sex') and inst.sex is not None else -1
            else:
                pt_id = f"sample_{idx}"
                age = np.nan
                sex = -1
                
            filename = f"{split_name}_{idx}.png"
            
            # 1. Extract Raw Modalities
            g = X_dict['g'].numpy()
            h = X_dict['h'].numpy()
            f = X_dict['f'].numpy()
            
            # 2. PCA Transform
            g_pc = pcas['g'].transform(g)
            h_pc = pcas['h'].transform(h)
            f_pc = pcas['f'].transform(f)
            
            # 3. Concatenate (T, 3) -> [Gaze, Head, Face]
            combined_pc = np.hstack([g_pc, h_pc, f_pc])
            
            # 4. Normalize
            combined_norm = scaler.fit_transform(combined_pc)
            
            # 5. Compute GASF
            channels = [compute_gasf(combined_norm[:, c]) for c in range(3)]
            
            # 6. Stack & Resize
            img_array = np.stack(channels, axis=-1)
            img_uint8 = ((img_array + 1) / 2 * 255).astype(np.uint8)
            img_resized = cv2.resize(img_uint8, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
            
            # 7. Save
            cv2.imwrite(os.path.join(save_dir, filename), img_resized)
            
            metadata_records.append({
                'filename': filename,
                'split': split_name,
                'pt_id': pt_id,
                'age': age,
                'sex': sex,
                'label': int(y.item())
            })
            
        except Exception as e:
            print(f"⚠️ Error processing sample {idx} in {split_name}: {e}")
            continue
        
    return pcas, metadata_records

if __name__ == "__main__":    
    print("Loading Datasets...")
    train_ds = BoaOpenFaceDataset.load_dataset(settings.get_dataset_path(DATASET_TYPE, settings.training_filename))
    val_ds = BoaOpenFaceDataset.load_dataset(settings.get_dataset_path(DATASET_TYPE, settings.validation_filename))
    test_ds = BoaOpenFaceDataset.load_dataset(settings.get_dataset_path(DATASET_TYPE, settings.test_filename))
    
    # =========================================================================
    # CRITICAL FIX: Manually inject stats into ALL datasets to prevent crash
    # =========================================================================
    print("🚑 Patching Dataset Statistics...")
    
    # 1. Define dummy stats (mean=0, std=1)
    # This effectively disables z-scoring on trial_id but keeps code running
    SAFE_STATS = (0.0, 1.0)
    
    datasets = {'train': train_ds, 'val': val_ds, 'test': test_ds}
    
    for name, ds in datasets.items():
        # Force set modalities
        ds.modalities = ['g', 'h', 'f']
        
        # Check and Force Set Stats
        current_stats = getattr(ds, 'trial_id_stats', None)
        if current_stats is None:
            print(f"   -> Fixing missing stats for {name} set.")
            ds.trial_id_stats = SAFE_STATS
        else:
            print(f"   -> {name} set has stats: {current_stats}")
            
    print("✅ Datasets patched.\n")
    # =========================================================================

    # Process
    pcas, meta_train = process_split(train_ds, "train", None)
    _, meta_val = process_split(val_ds, "val", pcas)
    _, meta_test = process_split(test_ds, "test", pcas)
    
    # Save CSV
    all_meta = meta_train + meta_val + meta_test
    df = pd.DataFrame(all_meta)
    df.to_csv(os.path.join(OUTPUT_BASE_DIR, "metadata.csv"), index=False)
    
    print(f"\n✅ Multimodal Image Generation Complete!")
    print(f"Output: {OUTPUT_BASE_DIR}")