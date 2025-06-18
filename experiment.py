import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def load_and_plot_results(results_folder_path):
    """
    Load saved results and create CV and test performance histograms
    
    Args:
        results_folder_path: Path to the folder containing the saved results
    """
    
    # Look for the results file
    results_file = os.path.join(results_folder_path, "all_results.pth")
    
    if not os.path.exists(results_file):
        print(f"Results file not found at: {results_file}")
        print("Looking for other possible files...")
        
        # List all .pth files in the directory
        pth_files = [f for f in os.listdir(results_folder_path) if f.endswith('.pth')]
        print(f"Found .pth files: {pth_files}")
        
        if 'experimental_setup.pth' in pth_files:
            results_file = os.path.join(results_folder_path, "experimental_setup.pth")
            print(f"Trying: {results_file}")
        else:
            print("Please specify the correct .pth file")
            return
    
    # Load the results
    print(f"Loading results from: {results_file}")
    try:
        results = torch.load(results_file, map_location='cpu')
        print("Results loaded successfully!")
        
        # Print available keys to understand the structure
        print(f"Available keys in results: {list(results.keys())}")
        
    except Exception as e:
        print(f"Error loading results: {e}")
        return
    
    # Extract data based on the structure
    try:
        # Try the improved CV design structure first
        if 'cv_summary' in results and 'generalization_results' in results:
            cv_summary = results['cv_summary']
            generalization_results = results['generalization_results']
            
        # Try alternative structure
        elif 'cv_results' in results and 'generalization_results' in results:
            # Need to calculate summary from cv_results
            cv_results = results['cv_results'] 
            generalization_results = results['generalization_results']
            
            # Calculate CV summary
            cv_summary = {}
            for loss_name in cv_results:
                metrics_summary = {}
                for metric in ['iou', 'f1', 'precision', 'recall']:
                    values = [fold_metrics[metric] for fold_metrics in cv_results[loss_name]]
                    metrics_summary[f'{metric}_mean'] = np.mean(values)
                    metrics_summary[f'{metric}_std'] = np.std(values)
                cv_summary[loss_name] = metrics_summary
                
        else:
            print("Could not find expected data structure in results file")
            print("Please check if this is the correct results file")
            return
            
    except Exception as e:
        print(f"Error extracting data: {e}")
        return
    
    # Create the plots
    create_performance_plots(cv_summary, generalization_results, results_folder_path)

def create_performance_plots(cv_summary, generalization_results, save_path):
    """
    Create CV and test performance histogram plots
    """
    
    # Extract data
    loss_names = list(cv_summary.keys())
    
    # Clean up loss names for display
    display_names = [name.replace('_loss', '').replace('_', ' ') for name in loss_names]
    
    # Get CV data
    cv_ious = [cv_summary[name]['iou_mean'] for name in loss_names]
    cv_stds = [cv_summary[name]['iou_std'] for name in loss_names]
    
    # Get test data  
    test_ious = [generalization_results[name]['test_metrics']['iou'] for name in loss_names]
    
    # Create colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(loss_names)))
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: CV Performance with std
    bars1 = ax1.bar(range(len(display_names)), cv_ious, yerr=cv_stds, 
                    capsize=5, color=colors, alpha=0.7, error_kw={'linewidth': 2})
    
    ax1.set_title('CV Performance (with std)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('IoU', fontsize=12)
    ax1.set_xticks(range(len(display_names)))
    ax1.set_xticklabels(display_names, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, max(cv_ious) * 1.2)
    
    # Plot 2: Test Performance (sorted by performance)
    # Sort for better visualization
    sorted_indices = np.argsort(test_ious)[::-1]
    sorted_names = [display_names[i] for i in sorted_indices]
    sorted_test_ious = [test_ious[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]
    
    bars2 = ax2.bar(range(len(sorted_names)), sorted_test_ious, 
                   color=sorted_colors, alpha=0.7)
    
    ax2.set_title('Test Performance (Generalization)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('IoU', fontsize=12)
    ax2.set_xticks(range(len(sorted_names)))
    ax2.set_xticklabels(sorted_names, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, max(test_ious) * 1.1)
    
    # Add value labels on bars
    for bar, iou in zip(bars2, sorted_test_ious):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{iou:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save the plot
    save_file = os.path.join(save_path, "cv_test_performance_comparison.png")
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_file}")
    
    plt.show()
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    print("\nCross-Validation Results (ranked by mean IoU):")
    cv_ranked = sorted(zip(loss_names, cv_ious, cv_stds), key=lambda x: x[1], reverse=True)
    for i, (name, mean_iou, std_iou) in enumerate(cv_ranked):
        display_name = name.replace('_loss', '').replace('_', ' ')
        print(f"{i+1:2d}. {display_name:20}: {mean_iou:.4f} ± {std_iou:.4f}")
    
    print("\nTest Set Results (ranked by IoU):")
    test_ranked = sorted(zip(loss_names, test_ious), key=lambda x: x[1], reverse=True)
    for i, (name, test_iou) in enumerate(test_ranked):
        display_name = name.replace('_loss', '').replace('_', ' ')
        print(f"{i+1:2d}. {display_name:20}: {test_iou:.4f}")
    
    # Check consistency
    cv_winner = cv_ranked[0][0]
    test_winner = test_ranked[0][0]
    
    print(f"\nModel Selection Summary:")
    print(f"CV Winner:   {cv_winner.replace('_loss', '').replace('_', ' ')}")
    print(f"Test Winner: {test_winner.replace('_loss', '').replace('_', ' ')}")
    
    if cv_winner == test_winner:
        print("✅ Consistent winner - high confidence in model selection!")
    else:
        print("⚠️  Different winners - follow CV for model selection")

def find_recent_results_folder(base_path="experiments"):
    """
    Find the most recent results folder
    """
    if not os.path.exists(base_path):
        print(f"Base path {base_path} does not exist")
        return None
    
    # Look for folders with loss comparison patterns
    folders = [f for f in os.listdir(base_path) 
              if os.path.isdir(os.path.join(base_path, f)) and 
              ('loss_comparison' in f or 'improved_loss' in f)]
    
    if not folders:
        print(f"No loss comparison folders found in {base_path}")
        return None
    
    # Sort by modification time (most recent first)
    folders_with_time = []
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        mod_time = os.path.getmtime(folder_path)
        folders_with_time.append((mod_time, folder, folder_path))
    
    folders_with_time.sort(reverse=True)
    
    print("Found results folders:")
    for i, (mod_time, folder, full_path) in enumerate(folders_with_time):
        print(f"{i+1}. {folder}")
    
    return folders_with_time[0][2]  # Return most recent folder path

def main():
    """
    Main function to load and plot results
    """
    print("Loss Function Performance Plotter")
    print("="*50)
    
    # Option 2: Manual path specification
    print("\nOption 2: Specify folder path manually")
    folder_path = input("Enter the path to your results folder: ").strip()
    
    if folder_path and os.path.exists(folder_path):
        load_and_plot_results(folder_path)
    else:
        print("Invalid folder path. Please check and try again.")

# Alternative: Direct usage if you know the path
def plot_from_path(results_folder_path):
    """
    Direct function to plot from a known path
    
    Usage:
    plot_from_path("experiments/improved_loss_comparison_20241208_143022")
    """
    load_and_plot_results(results_folder_path)

if __name__ == "__main__":
    main()

# Example usage:
# If you know your exact folder path, you can use:
# plot_from_path("experiments/your_folder_name_here")