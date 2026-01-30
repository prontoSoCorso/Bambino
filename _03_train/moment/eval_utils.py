import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                             balanced_accuracy_score, matthews_corrcoef, 
                             f1_score, brier_score_loss, ConfusionMatrixDisplay)
from sklearn.calibration import calibration_curve

def evaluate_and_plot(y_true, y_pred, y_probs, meta_df, output_dir, model_name="Model"):
    """
    Generates all standard evaluation metrics and plots.
    meta_df must contain ['pt_id', 'age', 'sex'] aligned with y_true.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Metrics
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    brier = brier_score_loss(y_true, y_probs)
    
    print(f"\n--- {model_name} Final Results ---")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    
    # Save Report
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, digits=4))
        f.write(f"\nBalanced Acc: {bal_acc:.4f}\nMCC: {mcc:.4f}\nBrier: {brier:.4f}")

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Control", "Stimulus"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # 3. Calibration Curve
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
    plt.figure(figsize=(6,6))
    plt.plot(prob_pred, prob_true, marker='o', label=f'{model_name} (Brier={brier:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "calibration_curve.png"))
    plt.close()

    # 4. Per-Baby Analysis (Age vs balanced accuracy)
    # Re-align metadata
    meta_df = meta_df.copy().reset_index(drop=True)
    meta_df['true'] = y_true
    meta_df['pred'] = y_pred
    
    baby_stats = []
    for pid in meta_df['pt_id'].unique():
        sub = meta_df[meta_df['pt_id'] == pid]
        if len(sub) < 1: continue
        bal_acc = balanced_accuracy_score(sub['true'], sub['pred'])
        age = sub['age'].iloc[0]
        baby_stats.append({'pt_id': pid, 'age': age, 'balanced_accuracy': bal_acc,
                           'f1': f1_score(sub['true'], sub['pred'])})
    
    stats_df = pd.DataFrame(baby_stats)
    
    # Scatter Plot
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=stats_df, x='age', y='balanced_accuracy', s=100, color='purple', edgecolor='black')
    sns.regplot(data=stats_df, x='age', y='balanced_accuracy', scatter=False, color='gray', line_kws={'linestyle':'--'})
    plt.title(f"{model_name}: Age vs. Balanced Accuracy")
    plt.xlabel("Age (Months)")
    plt.ylabel("Balanced Accuracy")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "age_correlation.png"))
    plt.close()
    
    print(f"✅ Plots saved to {output_dir}")