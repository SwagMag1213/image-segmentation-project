import os
import torch
import pandas as pd
import numpy as np

def load_results_safe(results_file):
    """Safe loading function for PyTorch 2.6"""
    try:
        results = torch.load(results_file, map_location='cpu', weights_only=False)
        return results
    except Exception:
        try:
            from collections import defaultdict
            torch.serialization.add_safe_globals([defaultdict])
            results = torch.load(results_file, map_location='cpu', weights_only=True)
            return results
        except Exception as e:
            print(f"❌ Failed to load {results_file}: {e}")
            return None

def extract_experiment_info(results):
    """Extract loss function and hyperparameter info from results"""
    if 'configs' not in results or not results['configs']:
        return "unknown_experiment"
    
    config = results['configs'][0]  # Use first config to identify experiment
    loss_fn = config.get('loss_fn', 'unknown')
    
    # Create descriptive experiment name
    if loss_fn == 'focal':
        alpha = config.get('focal_alpha', 0.25)
        gamma = config.get('focal_gamma', 2.0)
        return f"Focal (α={alpha}, γ={gamma})"
    
    elif loss_fn == 'combo':
        alpha = config.get('loss_alpha', 0.5)
        return f"Combo (α={alpha})"
    
    elif loss_fn == 'dice':
        smooth = config.get('smooth', 1.0)
        return f"Dice (smooth={smooth})"
    
    elif 'tversky' in loss_fn:
        alpha = config.get('tversky_alpha', 0.5)
        beta = config.get('tversky_beta', 0.5)
        return f"Tversky (α={alpha}, β={beta})"
    
    elif loss_fn == 'boundary':
        return "Boundary"
    
    elif loss_fn == 'bce':
        return "BCE"
    
    else:
        return loss_fn.replace('_', ' ').title()

def create_performance_table(experiment_folders, save_csv=True, output_name="performance_comparison"):
    """
    Create a simple performance comparison table
    
    Args:
        experiment_folders: List of folder paths containing all_results.pth
        save_csv: Whether to save results as CSV
        output_name: Name for output files
    """
    
    print("="*80)
    print("PERFORMANCE COMPARISON TABLE")
    print("="*80)
    
    # Storage for table data
    table_data = []
    
    # Process each experiment
    for i, folder in enumerate(experiment_folders):
        results_file = os.path.join(folder, "all_results.pth")
        
        if not os.path.exists(results_file):
            print(f"⚠️  Skipping {folder} - no all_results.pth found")
            continue
        
        print(f"📂 Loading: {folder}")
        results = load_results_safe(results_file)
        
        if results is None:
            continue
        
        # Extract experiment info
        experiment_name = extract_experiment_info(results)
        print(f"   Experiment: {experiment_name}")
        
        # Get CV and test results
        cv_summary = results.get('cv_summary', {})
        generalization_results = results.get('generalization_results', {})
        
        # Process each configuration in this experiment
        for config_name in cv_summary.keys():
            if config_name not in generalization_results:
                continue
            
            cv_data = cv_summary[config_name]
            test_data = generalization_results[config_name]['test_metrics']
            
            # Extract model configuration details
            config_info = None
            if 'configs' in results:
                for config in results['configs']:
                    if config['name'] == config_name:
                        config_info = config
                        break
            
            # Parse configuration name for model details
            backbone = config_info.get('backbone', 'unknown') if config_info else 'unknown'
            attention = config_info.get('use_attention', False) if config_info else False
            batch_size = config_info.get('batch_size', 'unknown') if config_info else 'unknown'
            
            # Create row for table
            row = {
                'Experiment': experiment_name,
                'Backbone': backbone,
                'Attention': 'Yes' if attention else 'No',
                'Batch_Size': batch_size,
                'CV_IoU_Mean': cv_data['iou_mean'],
                'CV_IoU_Std': cv_data['iou_std'],
                'CV_F1_Mean': cv_data['f1_mean'],
                'CV_F1_Std': cv_data['f1_std'],
                'Test_IoU': test_data['iou'],
                'Test_F1': test_data['f1'],
                'Test_Precision': test_data['precision'],
                'Test_Recall': test_data['recall'],
                'CV_Test_Gap': cv_data['iou_mean'] - test_data['iou']  # Overfitting indicator
            }
            
            table_data.append(row)
        
        print(f"   Added {len(cv_summary)} configurations")
    
    if not table_data:
        print("❌ No data found to compare!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Sort by test IoU (best to worst)
    df = df.sort_values('Test_IoU', ascending=False).reset_index(drop=True)
    
    # Add ranking
    df.insert(0, 'Rank', range(1, len(df) + 1))
    
    print(f"\n✅ Created table with {len(df)} configurations from {len(set(df['Experiment']))} experiments")
    
    return df

def display_summary_table(df, top_n=10):
    """Display a clean summary table"""
    
    print(f"\n{'='*120}")
    print(f"TOP {min(top_n, len(df))} CONFIGURATIONS (by Test IoU)")
    print(f"{'='*120}")
    
    # Create a formatted display table
    display_df = df.head(top_n).copy()
    
    # Format columns for better display
    display_df['CV_IoU'] = display_df.apply(lambda x: f"{x['CV_IoU_Mean']:.3f}±{x['CV_IoU_Std']:.3f}", axis=1)
    display_df['CV_F1'] = display_df.apply(lambda x: f"{x['CV_F1_Mean']:.3f}±{x['CV_F1_Std']:.3f}", axis=1)
    display_df['Test_IoU'] = display_df['Test_IoU'].apply(lambda x: f"{x:.3f}")
    display_df['Test_F1'] = display_df['Test_F1'].apply(lambda x: f"{x:.3f}")
    display_df['Gap'] = display_df['CV_Test_Gap'].apply(lambda x: f"{x:+.3f}")
    
    # Select columns for display
    summary_cols = ['Rank', 'Experiment', 'Backbone', 'Attention', 'Batch_Size', 
                   'CV_IoU', 'Test_IoU', 'CV_F1', 'Test_F1', 'Gap']
    
    display_table = display_df[summary_cols]
    
    # Print with proper formatting
    print(display_table.to_string(index=False, max_colwidth=15))
    
    print(f"\n{'='*120}")
    print("LEGEND:")
    print("  Gap: CV_IoU - Test_IoU (negative = good generalization, positive = overfitting)")
    print("  CV_IoU: Cross-validation IoU (mean ± std across 5 folds)")
    print("  Test_IoU: Final test set performance (generalization)")

def experiment_summary(df):
    """Create experiment-level summary"""
    
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY (Average Performance)")
    print(f"{'='*80}")
    
    # Group by experiment and calculate averages
    exp_summary = df.groupby('Experiment').agg({
        'CV_IoU_Mean': ['mean', 'std'],
        'Test_IoU': ['mean', 'std'],
        'Test_F1': ['mean', 'std'],
        'CV_Test_Gap': 'mean'
    }).round(4)
    
    # Flatten column names
    exp_summary.columns = ['CV_IoU_Avg', 'CV_IoU_StdDev', 'Test_IoU_Avg', 'Test_IoU_StdDev', 
                          'Test_F1_Avg', 'Test_F1_StdDev', 'Avg_Gap']
    
    # Sort by average test IoU
    exp_summary = exp_summary.sort_values('Test_IoU_Avg', ascending=False)
    
    # Add configuration count
    config_counts = df.groupby('Experiment').size()
    exp_summary['Configs'] = config_counts
    
    # Format for display
    print(f"{'Experiment':<25} {'Configs':<7} {'CV IoU':<12} {'Test IoU':<12} {'Test F1':<12} {'Gap':<8}")
    print("-" * 80)
    
    for exp_name, row in exp_summary.iterrows():
        cv_str = f"{row['CV_IoU_Avg']:.3f}±{row['CV_IoU_StdDev']:.3f}"
        test_iou_str = f"{row['Test_IoU_Avg']:.3f}±{row['Test_IoU_StdDev']:.3f}"
        test_f1_str = f"{row['Test_F1_Avg']:.3f}±{row['Test_F1_StdDev']:.3f}"
        gap_str = f"{row['Avg_Gap']:+.3f}"
        
        #print(f"{exp_name:<25} {row['Configs']:<7d} {cv_str:<12} {test_iou_str:<12} {test_f1_str:<12} {gap_str:<8}")
    
    # Find winner
    best_exp = exp_summary.index[0]
    best_avg = exp_summary.iloc[0]['Test_IoU_Avg']
    
    print(f"\n🏆 WINNER (by average): {best_exp} (Avg Test IoU: {best_avg:.3f})")
    
    return exp_summary

def main():
    """Main function to create performance comparison table"""
    
    print("Performance Table Generator")
    print("="*40)
    
    # 👇 UPDATE THESE PATHS TO YOUR ACTUAL EXPERIMENT FOLDERS
    experiment_folders = [
        "experiments/combo_alpha_05",  # Update these!
        "experiments/combo_alpha_025",
        "experiments/combo_alpha_075",
        "experiments/dice_smooth_1",
        "experiments/focal_alpha_05_gamma_3",
        "experiments/focal_alpha_075_gamma_2",
        "experiments/focal_alpha_025_gamma_2",
    ]
    
    # Create performance table
    df = create_performance_table(experiment_folders)
    
    if df is None:
        return
    
    # Display top results
    display_summary_table(df, top_n=10)
    
    # Show experiment summary
    exp_summary = experiment_summary(df)
    
    # Save to CSV
    output_name = f"performance_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Save detailed results
    csv_file = f"{output_name}_detailed.csv"
    df.to_csv(csv_file, index=False)
    print(f"\n💾 Detailed results saved: {csv_file}")
    
    # Save experiment summary
    summary_file = f"{output_name}_summary.csv"
    exp_summary.to_csv(summary_file)
    print(f"💾 Experiment summary saved: {summary_file}")
    
    # Print best configuration
    best_config = df.iloc[0]
    print(f"\n🎯 BEST OVERALL CONFIGURATION:")
    print(f"   Experiment: {best_config['Experiment']}")
    print(f"   Architecture: {best_config['Backbone']} + {'Attention' if best_config['Attention']=='Yes' else 'No Attention'}")
    print(f"   Batch Size: {best_config['Batch_Size']}")
    print(f"   CV IoU: {best_config['CV_IoU_Mean']:.3f} ± {best_config['CV_IoU_Std']:.3f}")
    print(f"   Test IoU: {best_config['Test_IoU']:.3f}")
    print(f"   Generalization: {best_config['CV_Test_Gap']:+.3f} (CV - Test)")

# Quick usage function
def quick_compare():
    """
    Quick function - just update paths and run
    """
    folders = [
        "experiments/combo_alpha_05",  # Update these!
        "experiments/combo_alpha_025",
        "experiments/combo_alpha_075",
        "experiments/dice_smooth_1",
        "experiments/focal_alpha_05_gamma_3",
        "experiments/focal_alpha075_gamma2",
    ]
    
    df = create_performance_table(folders)
    if df is not None:
        display_summary_table(df)
        experiment_summary(df)

if __name__ == "__main__":
    main()