"""
Loss Function Comparison using the modular framework
Uses cross_validation.py, train.py, and dataset.py modules
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

# Import from our modular framework
from cross_validation import ModelComparator
from advanced_models import UNetWithBackbone


def get_loss_configurations():
    """Define loss function configurations to test"""
    base_config = {
        'backbone': 'resnet34',
        'use_attention': False,
        'batch_size': 4,
        'num_epochs': 50,  # Reduced for testing - increase for real experiments
        'img_size': (128, 128),
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'pretrained': True,
        'early_stopping_patience': 5,
        'early_stopping_min_delta': 0.001,
        'verbose': True,
        'save_plots': False  # Disable individual plots during CV
    }
    
    # Define different loss function configurations
    # Organized by categories from the survey paper
    loss_configs = [
        # ============= Distribution-based Losses =============
        {**base_config, 'name': 'BCE Loss', 'loss_fn': 'bce'},
        {**base_config, 'name': 'Weighted BCE', 'loss_fn': 'weighted_bce', 'beta': 2.0},
        {**base_config, 'name': 'Balanced BCE', 'loss_fn': 'balanced_bce'},
        {**base_config, 'name': 'Focal Loss', 'loss_fn': 'focal', 
         'focal_alpha': 0.25, 'focal_gamma': 2.0},
        {**base_config, 'name': 'Distance Map BCE', 'loss_fn': 'distance_map_bce', 
         'distance_alpha': 1.0},
        
        # ============= Region-based Losses =============
        {**base_config, 'name': 'Dice Loss', 'loss_fn': 'dice', 'smooth': 1.0},
        {**base_config, 'name': 'Tversky Loss', 'loss_fn': 'tversky', 
         'tversky_alpha': 0.5, 'tversky_beta': 0.5},
        {**base_config, 'name': 'Tversky (Recall)', 'loss_fn': 'tversky_recall'},
        {**base_config, 'name': 'Focal Tversky', 'loss_fn': 'focal_tversky', 
         'tversky_alpha': 0.5, 'tversky_beta': 0.5, 'focal_tversky_gamma': 0.75},
        {**base_config, 'name': 'Sensitivity-Specificity', 'loss_fn': 'sensitivity_specificity', 
         'sensitivity_weight': 0.5},
        {**base_config, 'name': 'Log-Cosh Dice', 'loss_fn': 'log_cosh_dice', 'smooth': 1.0},
        
        # ============= Boundary-based Losses =============
        {**base_config, 'name': 'Boundary Loss', 'loss_fn': 'boundary', 
         'boundary_theta0': 3, 'boundary_theta': 5},
        
        # ============= Compound Losses =============
        {**base_config, 'name': 'Combo Loss', 'loss_fn': 'combo', 'loss_alpha': 0.5},
        {**base_config, 'name': 'Triple Combo', 'loss_fn': 'triple_combo',
         'alpha_dice': 0.33, 'alpha_bce': 0.33, 'alpha_focal': 0.34,
         'focal_alpha': 0.25, 'focal_gamma': 2.0},
    ]
    
    return loss_configs


def plot_cv_results(cv_results, save_dir):
    """Plot cross-validation results comparison"""
    # Extract data for plotting
    cv_summary = cv_results['comparison_summary']
    loss_names = list(cv_summary.keys())
    
    # Get metrics
    cv_ious_mean = [cv_summary[name]['iou_mean'] for name in loss_names]
    cv_ious_std = [cv_summary[name]['iou_std'] for name in loss_names]
    cv_f1_mean = [cv_summary[name]['f1_mean'] for name in loss_names]
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Plot 1: IoU comparison with error bars
    plt.subplot(1, 3, 1)
    sorted_idx = np.argsort(cv_ious_mean)[::-1]
    sorted_names = [loss_names[i] for i in sorted_idx]
    sorted_ious = [cv_ious_mean[i] for i in sorted_idx]
    sorted_stds = [cv_ious_std[i] for i in sorted_idx]
    
    bars = plt.bar(range(len(sorted_names)), sorted_ious, 
                    yerr=sorted_stds, capsize=5, alpha=0.7)
    plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha='right')
    plt.ylabel('IoU')
    plt.title('Cross-Validation IoU Performance')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, iou, std in zip(bars, sorted_ious, sorted_stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f'{iou:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: F1 Score comparison
    plt.subplot(1, 3, 2)
    sorted_f1 = [cv_f1_mean[i] for i in sorted_idx]
    bars = plt.bar(range(len(sorted_names)), sorted_f1, alpha=0.7, color='orange')
    plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha='right')
    plt.ylabel('F1 Score')
    plt.title('Cross-Validation F1 Performance')
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 3: Performance summary table
    plt.subplot(1, 3, 3)
    plt.axis('off')
    
    # Create table data
    table_data = []
    for i in sorted_idx:
        name = loss_names[i]
        table_data.append([
            name,
            f"{cv_ious_mean[i]:.4f} ± {cv_ious_std[i]:.4f}",
            f"{cv_f1_mean[i]:.4f}"
        ])
    
    table = plt.table(cellText=table_data,
                     colLabels=['Loss Function', 'CV IoU', 'CV F1'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    plt.title('Cross-Validation Summary\n(Ranked by IoU)', pad=20)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/cv_loss_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_generalization_results(cv_results, gen_results, save_dir):
    """Plot generalization test results and comparison with CV"""
    cv_summary = cv_results['comparison_summary']
    loss_names = list(gen_results.keys())
    
    # Extract metrics
    cv_ious = [cv_summary[name]['iou_mean'] for name in loss_names]
    test_ious = [gen_results[name]['test_metrics']['iou'] for name in loss_names]
    train_ious = [gen_results[name]['final_train_iou'] for name in loss_names]
    overfitting = [train_ious[i] - test_ious[i] for i in range(len(loss_names))]
    
    # Create figure
    plt.figure(figsize=(16, 10))
    
    # Plot 1: CV vs Test Performance scatter
    plt.subplot(2, 3, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(loss_names)))
    plt.scatter(cv_ious, test_ious, c=colors, s=150, alpha=0.7, edgecolors='black')
    
    for i, name in enumerate(loss_names):
        plt.annotate(name, (cv_ious[i], test_ious[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Add correlation line
    z = np.polyfit(cv_ious, test_ious, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(cv_ious), max(cv_ious), 100)
    plt.plot(x_line, p(x_line), "r--", alpha=0.8)
    
    correlation = np.corrcoef(cv_ious, test_ious)[0, 1]
    plt.xlabel('CV Mean IoU')
    plt.ylabel('Test IoU')
    plt.title(f'CV vs Test Performance\n(r = {correlation:.3f})')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Test Performance Ranking
    plt.subplot(2, 3, 2)
    sorted_test_idx = np.argsort(test_ious)[::-1]
    sorted_names = [loss_names[i] for i in sorted_test_idx]
    sorted_test_ious = [test_ious[i] for i in sorted_test_idx]
    
    bars = plt.bar(range(len(sorted_names)), sorted_test_ious, 
                    color=[colors[i] for i in sorted_test_idx], alpha=0.7)
    plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha='right')
    plt.ylabel('IoU')
    plt.title('Test Set Performance (Generalization)')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, iou in zip(bars, sorted_test_ious):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{iou:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Overfitting Analysis
    plt.subplot(2, 3, 3)
    sorted_overfit = [overfitting[i] for i in sorted_test_idx]
    bars = plt.bar(range(len(sorted_names)), sorted_overfit, 
                    color=[colors[i] for i in sorted_test_idx], alpha=0.7)
    plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha='right')
    plt.ylabel('Train IoU - Test IoU')
    plt.title('Overfitting Analysis')
    plt.grid(axis='y', alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Plot 4: Method Ranking Comparison
    plt.subplot(2, 3, 4)
    cv_sorted_idx = np.argsort(cv_ious)[::-1]
    cv_ranks = [cv_sorted_idx.tolist().index(i) + 1 for i in range(len(loss_names))]
    test_ranks = [sorted_test_idx.tolist().index(i) + 1 for i in range(len(loss_names))]
    
    for i, name in enumerate(loss_names):
        plt.plot([1, 2], [cv_ranks[i], test_ranks[i]], 'o-', 
                color=colors[i], alpha=0.7, linewidth=2, markersize=8)
        # Add labels for top 3
        if cv_ranks[i] <= 3 or test_ranks[i] <= 3:
            plt.text(1.5, (cv_ranks[i] + test_ranks[i])/2, name, 
                    ha='center', va='center', fontsize=8, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.5))
    
    plt.xticks([1, 2], ['CV Rank', 'Test Rank'])
    plt.ylabel('Rank (1 = best)')
    plt.title('Ranking Stability')
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()
    
    # Plot 5: Comprehensive Summary Table
    plt.subplot(2, 3, 5)
    plt.axis('off')
    
    table_data = []
    for i in sorted_test_idx:
        name = loss_names[i]
        cv_rank = cv_ranks[i]
        test_rank = test_ranks[i]
        rank_change = cv_rank - test_rank
        rank_symbol = '↑' if rank_change > 0 else ('↓' if rank_change < 0 else '=')
        
        table_data.append([
            name,
            f"{cv_ious[i]:.4f}",
            f"{test_ious[i]:.4f}",
            f"{overfitting[i]:.4f}",
            f"{cv_rank} → {test_rank} {rank_symbol}"
        ])
    
    table = plt.table(cellText=table_data,
                     colLabels=['Loss Function', 'CV IoU', 'Test IoU', 'Overfit', 'Rank Change'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    plt.title('Complete Performance Summary\n(Ranked by Test IoU)', pad=20)
    
    # Plot 6: Training curves comparison (for top 3)
    plt.subplot(2, 3, 6)
    plt.text(0.5, 0.5, 'Individual training curves\ncan be generated separately\nfor detailed analysis', 
             ha='center', va='center', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray'))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Training Analysis')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/generalization_loss_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main experiment runner using the modular framework"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"experiments/loss_comparison_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Configuration
    data_dir = "manual_labels"
    image_type = 'W'
    n_splits = 5
    test_size = 0.2
    augmentations_per_image = 3
    random_state = 42
    
    print("="*80)
    print("LOSS FUNCTION COMPARISON USING MODULAR FRAMEWORK")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Image type: {image_type}")
    print(f"Cross-validation folds: {n_splits}")
    print(f"Test set size: {test_size*100:.0f}%")
    print(f"Augmentations per image: {augmentations_per_image}")
    print(f"Results will be saved to: {save_dir}")
    print("="*80)
    
    # Get loss function configurations
    loss_configs = get_loss_configurations()
    print(f"\nTesting {len(loss_configs)} loss functions:")
    for config in loss_configs:
        print(f"  - {config['name']}")
    
    # Create model configurations (model class + loss config)
    model_configs = [(UNetWithBackbone, config) for config in loss_configs]
    
    # Initialize comparator (handles train/test split)
    print(f"\n{'='*60}")
    print("PHASE 1: CROSS-VALIDATION ON TRAINING SET")
    print(f"{'='*60}")
    
    comparator = ModelComparator(
        data_dir=data_dir,
        image_type=image_type,
        test_size=test_size,
        n_splits=n_splits,
        random_state=random_state,
        augmentations_per_image=augmentations_per_image,
        verbose=True
    )
    
    # Run cross-validation comparison
    cv_results = comparator.run_cv_comparison(model_configs)
    
    # Save CV results
    torch.save({
        'cv_results': cv_results,
        'configs': loss_configs,
        'timestamp': timestamp
    }, f"{save_dir}/cv_results.pth")
    
    # Plot CV results
    print("\nGenerating cross-validation comparison plots...")
    plot_cv_results(cv_results, save_dir)
    
    # Run generalization evaluation
    print(f"\n{'='*60}")
    print("PHASE 2: GENERALIZATION EVALUATION ON TEST SET")
    print(f"{'='*60}")
    
    gen_results = comparator.evaluate_generalization(model_configs)
    
    # Save generalization results
    torch.save({
        'generalization_results': gen_results,
        'configs': loss_configs,
        'timestamp': timestamp
    }, f"{save_dir}/generalization_results.pth")
    
    # Plot comprehensive results
    print("\nGenerating comprehensive comparison plots...")
    plot_generalization_results(cv_results, gen_results, save_dir)
    
    # Final summary
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    
    # Best by CV
    cv_summary = cv_results['comparison_summary']
    best_cv = max(cv_summary.items(), key=lambda x: x[1]['iou_mean'])
    print(f"\nBest by Cross-Validation:")
    print(f"  {best_cv[0]}: IoU = {best_cv[1]['iou_mean']:.4f} ± {best_cv[1]['iou_std']:.4f}")
    
    # Best by Test
    best_test = max(gen_results.items(), key=lambda x: x[1]['test_metrics']['iou'])
    print(f"\nBest by Test Set Performance:")
    print(f"  {best_test[0]}: IoU = {best_test[1]['test_metrics']['iou']:.4f}")
    
    # Check consistency
    if best_cv[0] == best_test[0]:
        print(f"\n✅ CONSISTENT WINNER: {best_cv[0]}")
        print("   Cross-validation successfully identified the best loss function!")
    else:
        print(f"\n⚠️  DIFFERENT WINNERS:")
        print(f"   CV Best: {best_cv[0]}")
        print(f"   Test Best: {best_test[0]}")
    
    # Calculate CV-Test correlation
    cv_ious = [cv_summary[name]['iou_mean'] for name in cv_summary.keys()]
    test_ious = [gen_results[name]['test_metrics']['iou'] for name in cv_summary.keys()]
    correlation = np.corrcoef(cv_ious, test_ious)[0, 1]
    
    print(f"\nCV-Test Correlation: r = {correlation:.3f}")
    if correlation > 0.8:
        print("   ✅ Strong correlation - CV is reliable for model selection")
    elif correlation > 0.5:
        print("   ⚠️  Moderate correlation - CV provides reasonable guidance")
    else:
        print("   ❌ Weak correlation - Consider other validation strategies")
    
    # Top 3 recommendations
    print(f"\n{'='*60}")
    print("TOP 3 LOSS FUNCTIONS (by test performance):")
    print(f"{'='*60}")
    sorted_test = sorted(gen_results.items(), key=lambda x: x[1]['test_metrics']['iou'], reverse=True)
    for i, (name, results) in enumerate(sorted_test[:3]):
        print(f"{i+1}. {name}:")
        print(f"   Test IoU: {results['test_metrics']['iou']:.4f}")
        print(f"   Test F1:  {results['test_metrics']['f1']:.4f}")
        print(f"   Overfitting: {results['final_train_iou'] - results['test_metrics']['iou']:.4f}")
    
    print(f"\nAll results saved to: {save_dir}")
    print("Experiment completed successfully!")


if __name__ == "__main__":
    main()