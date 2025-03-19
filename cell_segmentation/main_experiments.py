import os
import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse
from run_experiment import run_single_experiment
from experiment_categories import (
    get_backbone_experiments,
    get_attention_experiments,
    get_loss_function_experiments,
    get_optimizer_experiments,
    get_image_size_experiments,
    get_batch_size_experiments,
    get_image_type_experiments
)

def run_experiment_category(category_name, experiments, use_cv=False):
    """
    Run a category of experiments
    
    Args:
        category_name: Name of the experiment category
        experiments: List of experiment definitions
        use_cv: Whether to use cross-validation
    """
    results = []
    print(f"\n=== Running {category_name} Experiments ===\n")
    
    for exp in experiments:
        experiment_name = f"{category_name}_{exp['experiment']}"
        result = run_single_experiment(experiment_name, exp['params'], cv=use_cv)
        results.append(result)
    
    # Create results directory if it doesn't exist
    os.makedirs('experiment_results', exist_ok=True)
    
    # Save results to CSV
    results_df = pd.DataFrame([
        {
            'Experiment': r['experiment'],
            'IoU': r['iou'],
            'IoU_StdDev': r.get('iou_std', 0) if use_cv else None,
            **{k: str(v) for k, v in r['config'].items() if k in exp['params']}
        } 
        for r, exp in zip(results, experiments)
    ])
    
    results_df.to_csv(f"experiment_results/{category_name}_results.csv", index=False)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    if use_cv:
        # Plot with error bars for cross-validation
        plt.bar(
            results_df['Experiment'],
            results_df['IoU'],
            yerr=results_df['IoU_StdDev'],
            capsize=10
        )
    else:
        # Simple bar plot for single training runs
        plt.bar(results_df['Experiment'], results_df['IoU'])
    
    plt.title(f"{category_name} - IoU Comparison")
    plt.ylabel("IoU Score")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f"experiment_results/{category_name}_comparison.png", dpi=200)
    plt.close()
    
    return results_df

def parse_args():
    parser = argparse.ArgumentParser(description="Run cell segmentation experiments")
    parser.add_argument('--category', type=str, default='all',
                      choices=['all', 'backbone', 'attention', 'loss', 'optimizer', 
                               'image_size', 'batch_size', 'image_type'],
                      help="Which category of experiments to run")
    parser.add_argument('--cv', action='store_true', 
                      help="Use cross-validation for more robust results")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create a mapping of categories to their experiment functions
    category_map = {
        'backbone': get_backbone_experiments,
        'attention': get_attention_experiments,
        'loss': get_loss_function_experiments,
        'optimizer': get_optimizer_experiments,
        'image_size': get_image_size_experiments,
        'batch_size': get_batch_size_experiments,
        'image_type': get_image_type_experiments
    }
    
    all_results = []
    
    if args.category == 'all':
        # Run all categories
        for category_name, get_experiments in category_map.items():
            results_df = run_experiment_category(
                category_name, 
                get_experiments(), 
                use_cv=args.cv
            )
            all_results.append(results_df)
    else:
        # Run just the selected category
        if args.category in category_map:
            results_df = run_experiment_category(
                args.category, 
                category_map[args.category](), 
                use_cv=args.cv
            )
            all_results.append(results_df)
        else:
            print(f"Unknown category: {args.category}")
            return
    
    # Combine all results if we ran multiple categories
    if len(all_results) > 1:
        combined_df = pd.concat(all_results)
        combined_df.to_csv("experiment_results/all_results.csv", index=False)
        
        # Create a summary plot of the best configuration from each category
        best_configs = []
        for df in all_results:
            best_idx = df['IoU'].idxmax()
            best_configs.append(df.iloc[best_idx])
        
        best_df = pd.DataFrame(best_configs)
        
        plt.figure(figsize=(12, 6))
        plt.bar(best_df['Experiment'], best_df['IoU'])
        plt.title("Best Configuration from Each Category - IoU Comparison")
        plt.ylabel("IoU Score")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig("experiment_results/best_configurations.png", dpi=200)
        plt.close()
    
    print("\nAll experiments completed. Results saved to 'experiment_results' directory.")

if __name__ == "__main__":
    main()
