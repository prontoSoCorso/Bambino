import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import torch

# --- Style Configuration ---
def set_style():
    """Sets a professional, colorblind-friendly plotting style."""
    sns.set_theme(context="notebook", style="whitegrid")
    params = {
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 300
    }
    plt.rcParams.update(params)

def get_class_indices(labels, n_examples=3):
    """Helper to find indices for N examples of each class."""
    # labels are numpy array
    idx_stim = np.where(labels == 1)[0]
    idx_ctrl = np.where(labels == 0)[0]
    
    # Safe sampling: handle cases where we have fewer samples than n_examples
    if len(idx_stim) > 0:
        sel_stim = np.random.choice(idx_stim, min(len(idx_stim), n_examples), replace=False)
    else:
        sel_stim = np.array([], dtype=int)
        
    if len(idx_ctrl) > 0:
        sel_ctrl = np.random.choice(idx_ctrl, min(len(idx_ctrl), n_examples), replace=False)
    else:
        sel_ctrl = np.array([], dtype=int)
    
    return sel_stim, sel_ctrl

def plot_reconstruction_heatmaps(vis_data, output_dir, prefix=""):
    """
    Plots Heatmaps for all 38 channels. Robust to missing classes.
    """
    set_style()
    os.makedirs(output_dir, exist_ok=True)
    
    X_true = np.array(vis_data['true']) 
    X_pred = np.array(vis_data['pred'])
    labels = np.array(vis_data['label'])
    
    idx_stim, idx_ctrl = get_class_indices(labels, n_examples=3)
    
    # Combine what we found
    indices_to_plot = np.concatenate([idx_ctrl, idx_stim]).astype(int)
    
    if len(indices_to_plot) == 0:
        print(f"[{prefix}] Warning: No samples found for heatmap plotting.")
        return

    # Create Figure
    n_rows = len(indices_to_plot)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 3 * n_rows), constrained_layout=True, squeeze=False)
    
    for row_idx, data_idx in enumerate(indices_to_plot):
        orig = X_true[data_idx]
        recon = X_pred[data_idx]
        diff = np.abs(orig - recon)
        lbl = labels[data_idx]
        
        lbl_str = "Stimulus (Normal)" if lbl == 1 else "Control (Anomaly)"
        color_tag = "#0072B2" if lbl == 1 else "#D55E00" # Blue vs Orange
        
        # Color scaling (robust to flat signals)
        vmin = np.percentile(orig, 1) if np.std(orig) > 1e-5 else np.min(orig)
        vmax = np.percentile(orig, 99) if np.std(orig) > 1e-5 else np.max(orig)
        
        # 1. Original
        axes[row_idx, 0].imshow(orig, aspect='auto', cmap='cividis', vmin=vmin, vmax=vmax)
        axes[row_idx, 0].set_ylabel(f"{lbl_str}\nChannels", fontsize=12, fontweight='bold', color=color_tag)
        if row_idx == 0: axes[row_idx, 0].set_title("Original Input", fontsize=14)
        
        # 2. Recon
        axes[row_idx, 1].imshow(recon, aspect='auto', cmap='cividis', vmin=vmin, vmax=vmax)
        if row_idx == 0: axes[row_idx, 1].set_title("Reconstruction", fontsize=14)
        
        # 3. Error
        err_max = np.percentile(diff, 98) if np.max(diff) > 0 else 1.0
        axes[row_idx, 2].imshow(diff, aspect='auto', cmap='magma', vmin=0, vmax=err_max)
        if row_idx == 0: axes[row_idx, 2].set_title("|Error|", fontsize=14)
        
    fig.suptitle(f"{prefix}: Reconstruction Heatmaps", fontsize=16, y=1.02)
    plt.savefig(os.path.join(output_dir, f"{prefix}_heatmaps_summary.png"), bbox_inches='tight')
    plt.close()

def plot_error_distribution(mse_scores, labels, output_dir, prefix=""):
    """
    Violin plot. Robust to single-class data.
    """
    set_style()
    os.makedirs(output_dir, exist_ok=True)
    
    label_map = {1: "Stimulus (Normal)", 0: "Control (Anomaly)"}
    groups = [label_map[l] for l in labels]
    
    import pandas as pd
    df = pd.DataFrame({'MSE': mse_scores, 'Group': groups})
    
    if df.empty:
        return

    plt.figure(figsize=(10, 6))
    cb_palette = {"Stimulus (Normal)": "#0072B2", "Control (Anomaly)": "#D55E00"}
    
    # Check if we have both classes; if not, seaborn might warn, but it won't crash
    sns.violinplot(data=df, x='Group', y='MSE', palette=cb_palette, inner=None, alpha=0.3, linewidth=0)
    sns.boxplot(data=df, x='Group', y='MSE', palette=cb_palette, width=0.2, showfliers=False, boxprops={'zorder': 2})
    sns.stripplot(data=df, x='Group', y='MSE', color='black', alpha=0.4, size=3, jitter=True, zorder=3)
    
    plt.title(f"{prefix}: Reconstruction Error Distribution", fontsize=16)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
    plt.xlabel("") 
    
    plt.savefig(os.path.join(output_dir, f"{prefix}_error_distribution.png"), bbox_inches='tight')
    plt.close()

def plot_channel_examples(vis_data, output_dir, prefix=""):
    """
    Robust Line Plots. Dynamically creates rows based on available classes.
    """
    set_style()
    os.makedirs(output_dir, exist_ok=True)
    
    X_true = np.array(vis_data['true'])
    X_pred = np.array(vis_data['pred'])
    labels = np.array(vis_data['label'])
    
    idx_stim, idx_ctrl = get_class_indices(labels, n_examples=1)
    
    rows = []
    row_names = []
    colors = []
    
    # Only add Control row if we actually found a Control sample
    if len(idx_ctrl) > 0:
        rows.append(idx_ctrl[0])
        row_names.append("Control (Anomaly)")
        colors.append("#D55E00") # Orange
    else:
        print(f"[{prefix}] Warning: No Control samples found in visualization batch.")

    # Only add Stimulus row if we actually found a Stimulus sample
    if len(idx_stim) > 0:
        rows.append(idx_stim[0])
        row_names.append("Stimulus (Normal)")
        colors.append("#0072B2") # Blue
    else:
        print(f"[{prefix}] Warning: No Stimulus samples found in visualization batch.")
    
    if not rows:
        print(f"[{prefix}] No data available to plot lines.")
        return

    ch_indices = [0, 10, 20] 
    ch_names = ["Channel 0 (Gaze)", "Channel 10 (Head)", "Channel 20 (Face AU)"]
    
    # Dynamically size figure based on how many rows we actually have
    n_rows_plot = len(rows)
    fig, axes = plt.subplots(n_rows_plot, 3, figsize=(15, 4 * n_rows_plot), sharex=True, squeeze=False)
    
    for r, (data_idx, r_name, color) in enumerate(zip(rows, row_names, colors)):
        for c, (ch_idx, ch_name) in enumerate(zip(ch_indices, ch_names)):
            
            true_line = X_true[data_idx, ch_idx, :]
            pred_line = X_pred[data_idx, ch_idx, :]
            
            ax = axes[r, c]
            ax.plot(true_line, label='Original', color='black', alpha=0.6, linewidth=1.5)
            ax.plot(pred_line, label='Reconstruction', color=color, linestyle='--', linewidth=2)
            
            if r == 0: ax.set_title(ch_name, fontsize=12)
            if c == 0: ax.set_ylabel(f"{r_name}\nAmplitude", fontweight='bold', color=color)
            
            if r == 0 and c == 0: ax.legend()
            
    fig.suptitle(f"{prefix}: Detailed Channel Reconstruction", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_line_details.png"))
    plt.close()