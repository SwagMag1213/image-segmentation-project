import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_results_safe(results_file):
    """
    Safe loading function that handles PyTorch 2.6 compatibility
    """
    print(f"Loading results from: {results_file}")
    
    # Method 1: Try with weights_only=False (recommended for your own files)
    try:
        results = torch.load(results_file, map_location='cpu', weights_only=False)
        print("✅ Results loaded successfully!")
        return results
    except Exception as e1:
        print(f"Method 1 failed: {e1}")
    
    # Method 2: Try with safe globals
    try:
        torch.serialization.add_safe_globals([defaultdict])
        results = torch.load(results_file, map_location='cpu', weights_only=True)
        print("✅ Results loaded successfully with safe globals!")
        return results
    except Exception as e2:
        print(f"Method 2 failed: {e2}")
    
    # Method 3: Try old PyTorch style
    try:
        results = torch.load(results_file, map_location='cpu')
        print("✅ Results loaded successfully with legacy method!")
        return results
    except Exception as e3:
        print(f"All methods failed. Final error: {e3}")
        return None

def quick_plot_results(folder_path):
    """
    Quick function to plot results from your folder
    """
    # Find the results file
    results_file = os.path.join(folder_path, "all_results.pth")
    
    if not os.path.exists(results_file):
        print(f"Looking for alternative files in {folder_path}...")
        pth_files = [f for f in os.listdir(folder_path) if f.endswith('.pth')]
        print(f"Found: {pth_files}")
        
        if pth_files:
            results_file = os.path.join(folder_path, pth_files[0])
            print(f"Using: {results_file}")
        else:
            print("No .pth files found!")
            return
    
    # Load results
    results = load_results_safe(results_file)
    if results is None:
        return
    
    print(f"Available keys: {list(results.keys())}")
    
    # Extract data based on available structure
    try:
        if 'cv_summary' in results and 'generalization_results' in results:
            cv_summary = results['cv_summary']
            generalization_results = results['generalization_results']
        elif 'cv_results' in results and 'generalization_results' in results:
            print("Converting cv_results to cv_summary...")
            cv_results = results['cv_results']
            generalization_results = results['generalization_results']
            
            # Calculate summary
            cv_summary = {}
            for loss_name in cv_results:
                metrics = {}
                for metric in ['iou', 'f1', 'precision', 'recall']:
                    values = [fold[metric] for fold in cv_results[loss_name]]
                    metrics[f'{metric}_mean'] = np.mean(values)
                    metrics[f'{metric}_std'] = np.std(values)
                cv_summary[loss_name] = metrics
        else:
            print("Could not find expected data structure")
            print("Available keys:", list(results.keys()))
            if len(results.keys()) > 0:
                first_key = list(results.keys())[0]
                print(f"First key contents: {type(results[first_key])}")
                if hasattr(results[first_key], 'keys'):
                    print(f"  Sub-keys: {list(results[first_key].keys())}")
            return
    except Exception as e:
        print(f"Error extracting data: {e}")
        return
    
    # Create plots
    create_simple_plots(cv_summary, generalization_results, folder_path)

def create_simple_plots(cv_summary, generalization_results, save_path):
    """
    Create simple CV and test performance plots - both ordered from best to worst
    """
    # Extract data
    loss_names = list(cv_summary.keys())
    display_names = [name.replace('_loss', '').replace('_', ' ') for name in loss_names]
    
    cv_ious = [cv_summary[name]['iou_mean'] for name in loss_names]
    cv_stds = [cv_summary[name]['iou_std'] for name in loss_names]
    test_ious = [generalization_results[name]['test_metrics']['iou'] for name in loss_names]
    
    # Colors - consistent across both plots
    colors = plt.cm.Set3(np.linspace(0, 1, len(loss_names)))
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: CV Performance (sorted by CV performance)
    cv_sorted_indices = np.argsort(cv_ious)[::-1]  # Best to worst
    cv_sorted_names = [display_names[i] for i in cv_sorted_indices]
    cv_sorted_ious = [cv_ious[i] for i in cv_sorted_indices]
    cv_sorted_stds = [cv_stds[i] for i in cv_sorted_indices]
    cv_sorted_colors = [colors[i] for i in cv_sorted_indices]
    
    bars1 = ax1.bar(range(len(cv_sorted_names)), cv_sorted_ious, yerr=cv_sorted_stds, 
                    capsize=5, color=cv_sorted_colors, alpha=0.8, error_kw={'linewidth': 2})
    
    ax1.set_title('CV Performance (with std)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('IoU', fontsize=14)
    ax1.set_xticks(range(len(cv_sorted_names)))
    ax1.set_xticklabels(cv_sorted_names, rotation=45, ha='right', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, max(cv_ious) * 1.3)
    
    # Add value labels for CV plot
    for bar, iou, std in zip(bars1, cv_sorted_ious, cv_sorted_stds):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f'{iou:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Test Performance (sorted by test performance)
    test_sorted_indices = np.argsort(test_ious)[::-1]  # Best to worst
    test_sorted_names = [display_names[i] for i in test_sorted_indices]
    test_sorted_ious = [test_ious[i] for i in test_sorted_indices]
    test_sorted_colors = [colors[i] for i in test_sorted_indices]
    
    bars2 = ax2.bar(range(len(test_sorted_names)), test_sorted_ious, 
                   color=test_sorted_colors, alpha=0.8)
    
    ax2.set_title('Test Performance (Generalization)', fontsize=16, fontweight='bold')
    ax2.set_ylabel('IoU', fontsize=14)
    ax2.set_xticks(range(len(test_sorted_names)))
    ax2.set_xticklabels(test_sorted_names, rotation=45, ha='right', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, max(test_ious) * 1.15)
    
    # Add value labels for test plot
    for bar, iou in zip(bars2, test_sorted_ious):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{iou:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    save_file = os.path.join(save_path, "performance_histograms.png")
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"📊 Plot saved to: {save_file}")
    
    plt.show()
    
    # Print results (matching the sorted order shown in plots)
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY (ordered as shown in plots)")
    print(f"{'='*60}")
    
    print("\n📈 Cross-Validation Results (best to worst):")
    for i, name_idx in enumerate(cv_sorted_indices):
        name = display_names[name_idx]
        mean_iou = cv_ious[name_idx]
        std_iou = cv_stds[name_idx]
        print(f"{i+1:2d}. {name:20}: {mean_iou:.4f} ± {std_iou:.4f}")
    
    print("\n🎯 Test Set Results (best to worst):")
    for i, name_idx in enumerate(test_sorted_indices):
        name = display_names[name_idx]
        test_iou = test_ious[name_idx]
        print(f"{i+1:2d}. {name:20}: {test_iou:.4f}")
    
    # Winner comparison
    cv_winner = display_names[cv_sorted_indices[0]]
    test_winner = display_names[test_sorted_indices[0]]
    
    print(f"\n🏆 WINNERS:")
    print(f"   CV Best:   {cv_winner}")
    print(f"   Test Best: {test_winner}")
    
    if cv_winner == test_winner:
        print("   ✅ Same winner in both! High confidence.")
    else:
        print("   ⚠️  Different winners. Follow CV for model selection.")

# Direct usage function
def plot_my_results():
    """
    Easy function - just change the folder path to yours
    """
    # 👇 CHANGE THIS TO YOUR FOLDER PATH
    folder_path = "experiments/test"  # Update this!
    
    quick_plot_results(folder_path)

if __name__ == "__main__":
    # Option 1: Use the direct function (easiest)
    plot_my_results()
    
    # Option 2: Manual input
    # folder_path = input("Enter your results folder path: ").strip()
    # quick_plot_results(folder_path)