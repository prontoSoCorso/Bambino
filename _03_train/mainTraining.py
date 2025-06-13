import sys
import os
import time
import optuna
from pathlib import Path
from optuna.pruners import MedianPruner

# Configurazione dei percorsi
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = current_dir
while not os.path.basename(parent_dir) == "Bambino":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_03_train_with_optimization as conf, utils
from LSTMFCN.LSTMFCN_main import lstmfcn_main
from DataUtils.OpenFaceDataset import OpenFaceDataset
from DataUtils.BoaOpenFaceDataset import BoaOpenFaceDataset

"""
OPTUNA:
Terms:
    - Study: optimization based on an objective function
    - Trial: a single execution of the objective function
The goal of a study is to find out the optimal set of hyperparameter values through multiple trials (e.g., n_trials=100)
"""

def create_study(model_name):
    study_dir = Path(__file__).resolve().parent / "optuna_studies"
    study_dir.mkdir(parents=True, exist_ok=True)
    db_path = study_dir / f"{model_name}.db"
    storage_name = f"sqlite:///{db_path}"
    
    print(f"📊 Study storage: {storage_name}")
    print(f"💡 To view dashboard run: optuna-dashboard {storage_name}")
    
    # TO SEE THE DASHBOARD (e.g.): optuna-dashboard sqlite:///_03_train/optuna_studies/LSTMFCN.db
    
    pruner = MedianPruner(
        n_startup_trials=10,  # Wait n trials before pruning
        n_warmup_steps=70,    # Wait m epochs before evaluating
        interval_steps=10     # Check pruning every k epochs
    )

    return optuna.create_study(
        direction="maximize",
        study_name=f"{model_name}",
        storage=storage_name,
        load_if_exists=True,
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(seed=utils.seed)
    )

def optimize_lstmfcn():
    """Optimize LSTM-FCN hyperparameters using Optuna"""
    print("🔍 Starting LSTM-FCN hyperparameter optimization...")
    
    # Create study
    study = create_study(model_name="LSTMFCN")
    
    # Check if study has previous trials
    if len(study.trials) > 0:
        print(f"📈 Resuming study with {len(study.trials)} completed trials")
        print(f"🏆 Current best value: {study.best_value:.4f}")
        print(f"🎯 Current best params: {study.best_params}")
    
    # Define objective function
    def objective(trial):
        try:
            # Run LSTM-FCN training with trial parameters
            val_score = lstmfcn_main(
                trial=trial,
                run_test_evaluation=False  # Disable test eval during optimization
            )
            return val_score
        except optuna.exceptions.TrialPruned:
            # Trial was pruned, re-raise the exception
            raise
        except Exception as e:
            print(f"⚠️ Trial {trial.number} failed with error: {str(e)}")
            # Return a very low score for failed trials
            return 0.0
    
    # Run optimization
    try:
        study.optimize(objective, n_trials=conf.optuna_n_trials_LSTMFCN)
    except KeyboardInterrupt:
        print("\n⏹️ Optimization interrupted by user")
    
    # Print optimization results
    print(f"\n{'='*60}")
    print(f"🎉 LSTM-FCN Optimization Complete!")
    print(f"{'='*60}")
    print(f"🔥 Best trial: {study.best_trial.number}")
    print(f"🏆 Best value: {study.best_value:.4f}")
    print(f"🎯 Best parameters:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
        
    return study.best_params


def train_lstmfcn_with_defaults():
    """Train LSTM-FCN with default hyperparameters"""
    print("🚀 Training LSTM-FCN with default hyperparameters...")
    
    lstmfcn_main(
        train_path="", val_path="", test_path="",
        default_path=True,
        run_test_evaluation=True,
        # Default hyperparameters will be used from config
    )

def main(models_to_train=["LSTMFCN"], 
         optimize=conf.optimize_with_optuna):
    """Main training function with optional optimization"""
    
    start_time = time.time()
    
    print(f"🎯 Models to train: {models_to_train}")
    print(f"🔧 Optimization mode: {'ON' if optimize else 'OFF'}")
    print(f"🌱 Random seed: {utils.seed}")
    
    for model in models_to_train:
        model_start_time = time.time()
        
        if model.lower() == "lstmfcn":
            if optimize:
                print(f"\n{'='*60}")
                print(f"🔍 OPTIMIZING {model.upper()}")
                print(f"{'='*60}")
                
                # Run optimization
                best_params = optimize_lstmfcn()
                
                print(f"\n{'='*60}")
                print(f"🔄 RETRAINING {model.upper()} WITH BEST PARAMETERS")
                print(f"{'='*60}")
                
                # After optimization, retrain with best params and run test evaluation
                lstmfcn_main(
                    trial=None,
                    run_test_evaluation=True,  # Enable final test evaluation
                    **best_params
                )
                
            else:
                print(f"\n{'='*60}")
                print(f"🚀 TRAINING {model.upper()} WITH DEFAULT PARAMETERS")
                print(f"{'='*60}")
                
                # Train with default parameters
                train_lstmfcn_with_defaults()
        
        else:
            print(f"⚠️ Unknown model: {model}")
            continue
        
        model_time = time.time() - model_start_time
        print(f"⏱️ {model.upper()} completed in {model_time:.2f} seconds")
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"🏁 ALL MODELS COMPLETED")
    print(f"{'='*60}")
    print(f"⏱️ Total execution time: {total_time:.2f} seconds")
    
    # Print summary
    print(f"\n📋 SUMMARY:")
    print(f"   Models trained: {models_to_train}")
    print(f"   Optimization: {'Enabled' if optimize else 'Disabled'}")
    if optimize:
        print(f"   Trials per model: {conf.optuna_n_trials_LSTMFCN}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    utils.seed_everything(utils.seed)
    
    # Run main training pipeline
    main()