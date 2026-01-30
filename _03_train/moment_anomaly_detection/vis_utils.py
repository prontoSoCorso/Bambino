import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import config_ad as cfg

def plot_reconstruction_examples(vis_data, output_dir, prefix=""):
    """
    Plots the first channel of the first few samples.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    n_samples = len(vis_data['true'])
    
    for i in range(n_samples):
        true_sig = vis_data['true'][i] # [C, T]
        pred_sig = vis_data['pred'][i]
        lbl = vis_data['label'][i]
        label_str = "Stimulus (Normal)" if lbl == 1 else "Control (Anomaly)"
        
        # Plot just the first 3 channels to avoid clutter
        plt.figure(figsize=(12, 6))
        for c in range(3):
            plt.subplot(3, 1, c+1)
            plt.plot(true_sig[c], label='Original', color='black', alpha=0.7)
            plt.plot(pred_sig[c], label='Reconstruction', color='red', linestyle='--')
            plt.title(f"Sample {i} ({label_str}) - Channel {c}")
            if c == 0: plt.legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_sample_{i}_label_{lbl}.png"))
        plt.close()

def plot_boxplot_errors(mse_scores, labels, output_dir, prefix=""):
    """
    Box plot of Reconstruction Error vs Class.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    data = []
    for mse, lbl in zip(mse_scores, labels):
        lbl_str = "Stimulus (1)" if lbl == 1 else "Control (0)"
        data.append({'MSE': mse, 'Group': lbl_str})
        
    import pandas as pd
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='Group', y='MSE', palette={'Stimulus (1)': 'lightblue', 'Control (0)': 'salmon'})
    plt.title(f"{prefix} Reconstruction Error by Group")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"{prefix}_boxplot_mse.png"))
    plt.close()
    
    # Calculate simple separation stats
    stim_mse = df[df['Group']=='Stimulus (1)']['MSE']
    ctrl_mse = df[df['Group']=='Control (0)']['MSE']
    
    print(f"\n--- {prefix} Stats ---")
    print(f"Stimulus Mean MSE: {stim_mse.mean():.4f} (std: {stim_mse.std():.4f})")
    print(f"Control  Mean MSE: {ctrl_mse.mean():.4f} (std: {ctrl_mse.std():.4f})")