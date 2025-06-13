''' Configuration file for the project'''

import os
import torch
import random
import numpy as np

# Rileva il percorso della cartella "cellPIV" in modo dinamico
current_file_path = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(current_file_path)
while os.path.basename(PROJECT_ROOT) != "Bambino":
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

class utils:
    num_classes     = 2
    seed            = 2025
    device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multi_gpu       = torch.cuda.device_count() > 1  # Variabile per controllare l'uso di più GPU
    
    base_data_path  = os.path.join(PROJECT_ROOT, "data", "boa")
    train_path      = os.path.join(base_data_path, "training_set.pt")
    validation_path = os.path.join(base_data_path, "validation_set.pt")
    test_path       = os.path.join(base_data_path, "test_set.pt")

    @staticmethod
    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Config_03_train:
    output_model_base_dir = os.path.join(PROJECT_ROOT, "_04_test", "best_models")
    output_dir_plots = os.path.join(PROJECT_ROOT, "_03_train", "test_results_after_training")
    save_plots = True
    modality_dims = {
        "g": 8,
        "h": 13,
        "f": 17
        }   
    """
    All features: 
    "g": 8,
    "h": 13,
    "f": 17
    """
    num_features = sum(modality_dims.values())


    # ROCKET Hyperparameters
    rocket_kernels = 20000

    classifier = "RF"           # RF / LR / XGB
    rf_n_estimators=300
    rf_max_depth=50              # Limit depth
    rf_min_split=5       # Require n samples to split
    rf_max_features='sqrt'        # Use sqrt(n_features) per split
    rf_n_jobs=-1
    rf_class_weight='balanced'    # Handle class imbalance

    lr_max_iter=5000
    solver='saga'
    penalty='elasticnet'
    lr_l1_ratio=0.5
    lr_class_weight='balanced' # Uses y values to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))




    # LSTM-FCN Hyperparameters
    # Training
    lstmfcn_num_epochs      = 100
    lstmfcn_batch_size      = 16
    lstmfcn_learning_rate   = 3e-6
    lstmfcn_use_focal       = True

    # Early stopping
    lstmfcn_early_stopping_patience = lstmfcn_num_epochs

    # Scheduler: OneCycleLR
    lstmfcn_pct_start       = 0.15
    lstmfcn_max_lr          = 3e-3
    lstmfcn_anneal_strategy = "cos"

    # Model
    enc_hidden_dim      = 64
    dropout_enc         = 0.2
    n_heads             = 4
    dropout_attn        = 0.15

    lstm_hidden_dim     = 64
    lstm_layers         = 1
    bidirectional       = True
    dropout_lstm        = 0.3

    cnn_kernel_sizes    = "5,7,9"
    cnn_filter_sizes    = "128,128,128"
    dropout_cnn         = 0.1
    se_ratio            = 0.3

    dropout_classifier  = 0.4


class Config_03_train_with_optimization(Config_03_train):
    # Enable/disable test evaluation
    run_test_evaluation = True
    
    # Optuna optimization control
    optimize_with_optuna = True
    optuna_n_trials_LSTMFCN = 23  # Number of optimization trials
    
    # LSTM-FCN search space for Optuna
    @staticmethod
    def get_lstmfcn_search_space(trial):
        """Define the hyperparameter search space for LSTM-FCN optimization"""
        return {
            # Training hyperparameters
            'lstmfcn_batch_size': trial.suggest_categorical('lstmfcn_batch_size', [8, 16, 32]),
            'lstmfcn_learning_rate': trial.suggest_float('lstmfcn_learning_rate', 1e-6, 1e-3, log=True),
            'lstmfcn_max_lr': trial.suggest_float('lstmfcn_max_lr', 1e-4, 1e-2, log=True),
            'lstmfcn_pct_start': trial.suggest_float('lstmfcn_pct_start', 0.1, 0.5),
            
            # Model architecture
            'enc_hidden_dim': trial.suggest_categorical('enc_hidden_dim', [32, 64, 128, 256]),
            'dropout_enc': trial.suggest_float('dropout_enc', 0.1, 0.5),
            'n_heads': trial.suggest_categorical('n_heads', [2, 3, 4, 8]),
            'dropout_attn': trial.suggest_float('dropout_attn', 0.0, 0.3),
            
            # LSTM parameters
            'lstm_hidden_dim': trial.suggest_categorical('lstm_hidden_dim', [64, 128, 256, 512]),
            'lstm_layers': trial.suggest_int('lstm_layers', 1, 3),
            'bidirectional': trial.suggest_categorical('bidirectional', [True, False]),
            'dropout_lstm': trial.suggest_float('dropout_lstm', 0.1, 0.4),
            
            # CNN parameters
            'cnn_filter_sizes': trial.suggest_categorical('cnn_filter_sizes', [
                "64,128,64", "128,256,128", "128,128,128", "256,512,256"
            ]),
            'cnn_kernel_sizes': trial.suggest_categorical('cnn_kernel_sizes', [
                "3,5,7", "5,7,9", "7,5,3", "3,3,3", "5,5,5"
            ]),
            'dropout_cnn': trial.suggest_float('dropout_cnn', 0.0, 0.3),
            
            # Squeeze-and-Excitation and classifier
            'se_ratio': trial.suggest_float('se_ratio', 0.1, 0.5),
            'dropout_classifier': trial.suggest_float('dropout_classifier', 0.1, 0.5),
            
            # Loss function
            'lstmfcn_use_focal': trial.suggest_categorical('lstmfcn_use_focal', [True, False])
        }
    