"""
Utility script for analyzing Optuna optimization results
"""

import optuna
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import numpy as np

def load_study(model_name, study_dir=None):
    """Load an existing Optuna study"""
    if study_dir is None:
        study_dir = Path(__file__).resolve().parent / "optuna_studies"
    
    db_path = study_dir / f"{model_name}.db"
    storage_name = f"sqlite:///{db_path}"
    
    if not db_path.exists():
        print(f"❌ Study database not found: {db_path}")
        return None
    
    try:
        study = optuna.load_study(
            study_name=model_name,
            storage=storage_name
        )
        return study
    except Exception as e:
        print(f"❌ Error loading study: {e}")
        return None

def analyze_study(study, save_plots=True, output_dir=None):
    """Analyze and visualize Optuna study results"""
    
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "optuna_analysis" / study.study_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Analyzing study: {study.study_name}")
    print(f"📈 Number of trials: {len(study.trials)}")
    print(f"🏆 Best value: {study.best_value:.4f}")
    print(f"🎯 Best parameters:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    
    # Convert trials to DataFrame
    df = study.trials_dataframe()
    df_complete = df[df['state'] == 'COMPLETE']
    
    if len(df_complete) == 0:
        print("❌ No completed trials found")
        return
    
    print(f"\n📊 Statistics:")
    print(f"   Completed trials: {len(df_complete)}")
    print(f"   Failed trials: {len(df[df['state'] == 'FAIL'])}")
    print(f"   Pruned trials: {len(df[df['state'] == 'PRUNED'])}")
    print(f"   Mean objective value: {df_complete['value'].mean():.4f}")
    print(f"   Std objective value: {df_complete['value'].std():.4f}")
    
    if not save_plots:
        return
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Optimization history
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(df_complete['number'], df_complete['value'], 'b-', alpha=0.7, linewidth=1)
    plt.scatter(df_complete['number'], df_complete['value'], c=df_complete['value'], 
                cmap='viridis', s=30, alpha=0.8)
    plt.xlabel('Trial Number')
    plt.ylabel('Objective Value (F1 Score)')
    plt.title('Optimization History')
    plt.grid(True, alpha=0.3)
    plt.colorbar(label='F1 Score')
    
    plt.subplot(1, 2, 2)
    # Best value so far
    best_so_far = df_complete['value'].cummax()
    plt.plot(df_complete['number'], best_so_far, 'r-', linewidth=2, label='Best so far')
    plt.fill_between(df_complete['number'], best_so_far, alpha=0.3, color='red')
    plt.xlabel('Trial Number')
    plt.ylabel('Best Objective Value')
    plt.title('Best Value Progress')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'optimization_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Parameter importance
    try:
        importance = optuna.importance.get_param_importances(study)
        
        plt.figure(figsize=(10, 6))
        params = list(importance.keys())
        values = list(importance.values())
        
        plt.barh(params, values)
        plt.xlabel('Importance')
        plt.title('Hyperparameter Importance')
        plt.tight_layout()
        plt.savefig(output_dir / 'parameter_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n🔍 Parameter importance:")
        for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"   {param}: {imp:.4f}")
            
    except Exception as e:
        print(f"⚠️ Could not calculate parameter importance: {e}")
    
    # 3. Parameter distributions for top trials
    top_n = min(10, len(df_complete))
    top_trials = df_complete.nlargest(top_n, 'value')
    
    # Get parameter columns
    param_cols = [col for col in df_complete.columns if col.startswith('params_')]
    
    if len(param_cols) > 0:
        fig, axes = plt.subplots(len(param_cols), 1, figsize=(12, 3*len(param_cols)))
        if len(param_cols) == 1:
            axes = [axes]
        
        for i, param_col in enumerate(param_cols):
            param_name = param_col.replace('params_', '')
            
            # Check if parameter is numeric or categorical
            param_values = df_complete[param_col].dropna()
            
            if param_values.dtype in ['int64', 'float64']:
                # Numeric parameter - use scatter plot
                axes[i].scatter(param_values, df_complete.loc[param_values.index, 'value'], 
                               alpha=0.6, s=30)
                axes[i].scatter(top_trials[param_col], top_trials['value'], 
                               color='red', s=50, alpha=0.8, label=f'Top {top_n}')
                axes[i].set_xlabel(param_name)
                axes[i].set_ylabel('Objective Value')
            else:
                # Categorical parameter - use box plot
                df_plot = df_complete[[param_col, 'value']].dropna()
                if len(df_plot) > 0:
                    sns.boxplot(data=df_plot, x=param_col, y='value', ax=axes[i])
                    axes[i].set_xlabel(param_name)
                    axes[i].set_ylabel('Objective Value')
                    axes[i].tick_params(axis='x', rotation=45)
            
            axes[i].set_title(f'Parameter: {param_name}')
            axes[i].grid(True, alpha=0.3)
            if i == 0:
                axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'parameter_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Save detailed results
    results = {
        'study_name': study.study_name,
        'n_trials': len(study.trials),
        'n_completed': len(df_complete),
        'best_value': study.best_value,
        'best_params': study.best_params,
        'best_trial_number': study.best_trial.number,
        'statistics': {
            'mean_value': df_complete['value'].mean(),
            'std_value': df_complete['value'].std(),
            'min_value': df_complete['value'].min(),
            'max_value': df_complete['value'].max(),
            'median_value': df_complete['value'].median()
        }
    }
    
    # Save to JSON
    results_file = output_dir / 'optimization_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save top trials to CSV
    top_trials.to_csv(output_dir / f'top_{top_n}_trials.csv', index=False)
    
    print(f"\n💾 Results saved to: {output_dir}")
    print(f"   📊 Plots: optimization_history.png, parameter_importance.png, parameter_distributions.png")
    print(f"   📄 Data: optimization_results.json, top_{top_n}_trials.csv")

def compare_studies(study_names, study_dir=None):
    """Compare multiple Optuna studies"""
    if study_dir is None:
        study_dir = Path(__file__).resolve().parent / "optuna_studies"
    
    studies = {}
    for name in study_names:
        study = load_study(name, study_dir)
        if study is not None:
            studies[name] = study
    
    if len(studies) == 0:
        print("❌ No valid studies found")
        return
    
    print(f"📊 Comparing {len(studies)} studies:")
    print("-" * 60)
    
    for name, study in studies.items():
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        print(f"{name}:")
        print(f"   Trials: {len(study.trials)} (completed: {len(completed_trials)})")
        if len(completed_trials) > 0:
            print(f"   Best value: {study.best_value:.4f}")
            print(f"   Best trial: #{study.best_trial.number}")
        print()

def main():
    """Main analysis function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Optuna optimization results')
    parser.add_argument('--model', type=str, default='LSTMFCN', 
                       help='Model name to analyze')
    parser.add_argument('--no-plots', action='store_true', 
                       help='Skip plot generation')
    parser.add_argument('--output-dir', type=str, 
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Load and analyze study
    study = load_study(args.model)
    if study is not None:
        analyze_study(study, save_plots=not args.no_plots, output_dir=args.output_dir)
    else:
        print(f"❌ Could not load study for model: {args.model}")

if __name__ == "__main__":
    main()