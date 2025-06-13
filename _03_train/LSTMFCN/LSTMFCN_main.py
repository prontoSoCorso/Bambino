import os, sys
import numpy as np
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings
import optuna
warnings.filterwarnings('ignore')
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Aggiungo il percorso del progetto al sys.path
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "Bambino":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_03_train as conf, Config_03_train_with_optimization as opt_conf, utils
from DataUtils.OpenFaceDataset import OpenFaceDataset
from DataUtils.BoaOpenFaceDataset import BoaOpenFaceDataset
from _utils_ import dataset_utils, models_utils, plot_utils
from _03_train.LSTMFCN import LSTMFCN_train_and_eval, LSTMFCN_model

device = utils.device

def lstmfcn_main(train_path="", val_path="", test_path="", default_path=True,
                 save_plots=conf.save_plots,
                 output_dir_plots=conf.output_dir_plots,
                 output_model_base_dir=conf.output_model_base_dir,
                 trial=None,
                 run_test_evaluation=getattr(opt_conf, 'run_test_evaluation', False),
                 **kwargs):
    optimization_mode = trial is not None

    if optimization_mode:
        # Get hyperparameters from Optuna trial
        trial_params = opt_conf.get_lstmfcn_search_space(trial)
        hyperparams = {**trial_params, **kwargs}  # kwargs can override trial params if needed
    else:
        # Use default config values or provided kwargs
        hyperparams = {
            'lstmfcn_batch_size': kwargs.get('lstmfcn_batch_size', conf.lstmfcn_batch_size),
            'lstmfcn_learning_rate': kwargs.get('lstmfcn_learning_rate', conf.lstmfcn_learning_rate),
            'lstmfcn_max_lr': kwargs.get('lstmfcn_max_lr', conf.lstmfcn_max_lr),
            'lstmfcn_pct_start': kwargs.get('lstmfcn_pct_start', conf.lstmfcn_pct_start),
            'lstmfcn_use_focal': kwargs.get('lstmfcn_use_focal', conf.lstmfcn_use_focal),
            'enc_hidden_dim': kwargs.get('enc_hidden_dim', conf.enc_hidden_dim),
            'dropout_enc': kwargs.get('dropout_enc', conf.dropout_enc),
            'n_heads': kwargs.get('n_heads', conf.n_heads),
            'dropout_attn': kwargs.get('dropout_attn', conf.dropout_attn),
            'lstm_hidden_dim': kwargs.get('lstm_hidden_dim', conf.lstm_hidden_dim),
            'lstm_layers': kwargs.get('lstm_layers', conf.lstm_layers),
            'bidirectional': kwargs.get('bidirectional', conf.bidirectional),
            'dropout_lstm': kwargs.get('dropout_lstm', conf.dropout_lstm),
            'cnn_filter_sizes': kwargs.get('cnn_filter_sizes', conf.cnn_filter_sizes),
            'cnn_kernel_sizes': kwargs.get('cnn_kernel_sizes', conf.cnn_kernel_sizes),
            'dropout_cnn': kwargs.get('dropout_cnn', conf.dropout_cnn),
            'se_ratio': kwargs.get('se_ratio', conf.se_ratio),
            'dropout_classifier': kwargs.get('dropout_classifier', conf.dropout_classifier),
        }
    # Logging & dirs
    if not optimization_mode:
        models_utils.config_logging(log_dir="logs", log_filename="train_lstmfcn.log")
        os.makedirs(output_model_base_dir, exist_ok=True)
        os.makedirs(output_dir_plots, exist_ok=True)
        
        print("=" * 60)
        print("🔧 Initializing LSTM-FCN Training Pipeline")
        print("=" * 60)

    # Load datasets, selecting only the modalities defined in config.modality_dims
    if not optimization_mode:
        print("\n📂 Loading datasets...")
    datasets = {}
    selected_modalities = list(conf.modality_dims.keys())
    for split, path in [("train", utils.train_path), ("val", utils.validation_path), ("test", utils.test_path)]:
        print(f"Loading {split} set from {path} with modalities {selected_modalities}")
        datasets[split] = BoaOpenFaceDataset.load_dataset(path, modalities=selected_modalities)
    
    train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]
    
    # Compute statistics
    if not optimization_mode:
        print("\n📊 Computing dataset statistics...")
    train_ds.compute_statistics()
    val_ds.trial_id_stats = train_ds.trial_id_stats
    test_ds.trial_id_stats = train_ds.trial_id_stats

    # Data quality analysis (not in optimization mode)
    if not optimization_mode:
        dataset_utils.analyze_data_quality(datasets)

    # ─────────────── NORMALIZATION STEP ───────────────
    # Compute per‐feature mean/std on the TRAIN set, then apply to all splits
    if not optimization_mode:
        print("\n🔄 Computing normalization parameters on TRAIN set...")
    norm_params = dataset_utils.compute_normalization_params(datasets["train"])

    if not optimization_mode:
        print("🔄 Applying normalization to all splits (train/val/test)...")
    for split in ("train", "val", "test"):
        dataset_utils.apply_normalization(datasets[split], norm_params)
    # ────────────────────────────────────────────────────
    
    # Verify normalization effects (only in normal mode)
    if not optimization_mode:
        print("\n📊 Post-normalization statistics:")
        dataset_utils.analyze_data_quality(datasets)
    
    # Create balanced sampler
    if not optimization_mode:
        print("\n⚖️ Creating balanced sampler...")
    train_sampler = models_utils.get_balanced_sampler(train_ds, utils.num_classes)

    # Data loaders
    if not optimization_mode:
        print("\n🔄 Creating data loaders...")
    batch_size = hyperparams['lstmfcn_batch_size']
    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=min(8, os.cpu_count()), 
                              pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=min(4, os.cpu_count()),
                            pin_memory=True)
    test_loader = DataLoader(test_ds,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=min(4, os.cpu_count()),
                             pin_memory=True)

    # Model initialization
    if not optimization_mode:
        print(f"\n🧠 Initializing LSTM-FCN model on {device}...")
    model = LSTMFCN_model.LSTMFCN(
        modality_dims=conf.modality_dims,
        enc_hidden_dim=hyperparams['enc_hidden_dim'],
        dropout_enc=hyperparams['dropout_enc'],
        num_features=conf.num_features,
        n_heads=hyperparams['n_heads'],
        dropout_attn=hyperparams['dropout_attn'],
        lstm_hidden_dim=hyperparams['lstm_hidden_dim'],
        lstm_layers=hyperparams['lstm_layers'],
        bidirectional=hyperparams['bidirectional'],
        dropout_lstm=hyperparams['dropout_lstm'],
        cnn_filter_sizes=hyperparams['cnn_filter_sizes'],
        cnn_kernel_sizes=hyperparams['cnn_kernel_sizes'],
        dropout_cnn=hyperparams['dropout_cnn'],
        se_ratio=hyperparams['se_ratio'],
        dropout_classifier=hyperparams['dropout_classifier']
    ).to(device)

    # Model summary (only in normal mode)
    if not optimization_mode:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📏 Total parameters: {total_params:,}")
        print(f"🎯 Trainable parameters: {trainable_params:,}")
    
    # Multi-GPU support
    if utils.multi_gpu:
        model = nn.DataParallel(model)

    # Training + Validation
    if not optimization_mode:
        print(f"\n🏋️ Starting training...")
    
    # Create a temporary config object with the current hyperparameters
    temp_conf = type('TempConfig', (), {})()
    for key, value in vars(conf).items():
        if not key.startswith('_'):
            setattr(temp_conf, key, value)
    
    # Override with current hyperparameters
    for key, value in hyperparams.items():
        setattr(temp_conf, key, value)

    # Set the model path
    if optimization_mode:
        best_model_path = os.path.join(conf.output_model_base_dir, f"temp_trial_{trial.number}.pth")
    else:
        best_model_path = os.path.join(conf.output_model_base_dir, "best_lstmfcn.pth")

    # Train the model
    model, train_history = LSTMFCN_train_and_eval.train_model(
        model, train_loader, val_loader, temp_conf, best_model_path, 
        trial=trial if optimization_mode else None
    )

    # For optimization mode, return the best validation score
    if optimization_mode:
        # Clean up temporary model file
        try:
            if os.path.exists(best_model_path):
                os.remove(best_model_path)
        except:
            pass
        
        # Return the best validation metric for Optuna
        best_val_score = max([epoch_data.get('val_mcc', 0) for epoch_data in train_history])
        return best_val_score

    # ─── PLOT TRAINING HISTORY ─────────────────────────────────────────
    plot_save_path = os.path.join(conf.output_dir_plots, "training_history.png")
    os.makedirs(conf.output_dir_plots, exist_ok=True)
    plot_utils.plot_training_history(train_history, save_path=plot_save_path, figsize=(16, 10), dpi=150)
    print(f"✅ Saved training‐history figure to: {plot_save_path}")
    # ───────────────────────────────────────────────────────────────────

    # Reload best weights onto "model", find the best threshold and update the model
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("\n🔎 Finding best threshold on validation set (post‐training)...")
    best_thr = LSTMFCN_train_and_eval.find_best_threshold(model, val_loader, thresholds=np.linspace(0, 1, 101))
    model.best_threshold = best_thr
    print(f"🔖 Selected best_threshold = {best_thr:.2f}\n")
    checkpoint['best_threshold'] = best_thr
    torch.save(checkpoint, best_model_path)

    # Final test evaluation
    if run_test_evaluation:
        print(f"\n🎯 Final Test Evaluation...")
        y_true, y_prob, test_loss = LSTMFCN_train_and_eval.evaluate_model(model, test_loader, desc="Final Test")
        threshold = checkpoint.get('best_threshold', 0.5)
        y_pred = (np.array(y_prob) >= threshold).astype(int)
        test_metrics = models_utils.calculate_metrics(y_true, y_pred, y_prob)

        print(f"\n📊 FINAL TEST RESULTS")
        print("=" * 40)
        logging.info(f"=== FINAL TEST RESULTS - thr: {best_thr:.2f} ===")

        for metric_name, value in test_metrics.items():
            if metric_name not in ('fpr','tpr', 'conf_matrix'):
                logging.info(f"{metric_name:<20}: {value:.4f}")
        logging.info(f"'conf_matrix': {test_metrics['conf_matrix']}")

        # Save results and plots
        print(f"\n💾 Saving results and plots...")

        # 7) Plots (ROC + conf mat)
        if save_plots:
            os.makedirs(conf.output_dir_plots, exist_ok=True)
            plot_utils.save_confusion_matrix(
                test_metrics['conf_matrix'],
                os.path.join(conf.output_dir_plots, "cm_lstmfcn.png"),
                f"LSTM-FCN)")
            
            plot_utils.plot_roc_curve(
                test_metrics['fpr'], 
                test_metrics['tpr'],
                test_metrics['roc_auc'],
                os.path.join(conf.output_dir_plots,"roc_lstmfcn.png"))
        
    print(f"✅ Training completed successfully!")
    print(f"📁 Model saved to: {conf.output_model_base_dir}")
    print(f"📊 Plots saved to: {conf.output_dir_plots}")


if __name__ == "__main__":
    utils.seed_everything(utils.seed)
    lstmfcn_main()