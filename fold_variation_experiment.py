"""
CV Fold Variation Experiment using Modular Framework
Tests different numbers of CV folds to find optimal configuration
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict

# Import from modular framework
from cross_validation import CrossValidator
from advanced_models import UNetWithBackbone


class FoldVariationExperiment:
    """Test different numbers of CV folds using the modular framework."""
    
    def __init__(self, base_config: Dict, fold_options: List[int] = [2, 3, 4, 5, 6, 7]):
        """
        Args:
            base_config: Base configuration for models
            fold_options: Different numbers of CV folds to test
        """
        self.base_config = base_config
        self.fold_options = fold_options
        self.results = {}
        
    def run_fold_comparison(self) -> Dict:
        """
        Test different numbers of CV folds using the same model configuration.
        Uses CrossValidator with different n_splits values.
        """
        print(f"{'='*60}")
        print("CV FOLD VARIATION EXPERIMENT")
        print(f"{'='*60}")
        print(f"Model: {self.base_config['backbone']} UNet")
        print(f"Testing fold options: {self.fold_options}")
        print(f"Fixed augmentations per image: {self.base_config.get('augmentations_per_image', 3)}")
        
        # Results storage
        fold_results = {}
        detailed_results = {}
        
        for n_folds in self.fold_options:
            print(f"\n{'='*40}")
            print(f"TESTING {n_folds}-FOLD CROSS-VALIDATION")
            print(f"{'='*40}")
            
            # Create CrossValidator with this fold configuration
            cv = CrossValidator(
                data_dir=self.base_config['data_dir'],
                image_type=self.base_config['image_type'],
                n_splits=n_folds,  # This is the key parameter we're testing
                random_state=self.base_config['random_state'],
                augmentations_per_image=self.base_config.get('augmentations_per_image', 3),
                verbose=True
            )
            
            # Check if we have enough data for this many folds
            total_samples = len(cv.image_paths)
            min_samples_per_fold = total_samples // n_folds
            
            if min_samples_per_fold < 2:
                print(f"⚠️  Skipping {n_folds} folds: Not enough data (need at least 2 samples per fold)")
                continue
            
            print(f"Total samples: {total_samples}")
            print(f"Approximate samples per fold: {min_samples_per_fold}")
            
            # Run cross-validation for this fold configuration
            cv_results = cv.cross_validate_single_model(
                model_class=UNetWithBackbone,
                config=self.base_config
            )
            
            # Extract key metrics
            mean_iou = cv_results['cv_summary']['iou_mean']
            std_iou = cv_results['cv_summary']['iou_std']
            mean_f1 = cv_results['cv_summary']['f1_mean']
            std_f1 = cv_results['cv_summary']['f1_std']
            
            # Store results
            fold_results[n_folds] = {
                'mean_iou': mean_iou,
                'std_iou': std_iou,
                'mean_f1': mean_f1,
                'std_f1': std_f1,
                'fold_scores': [fold['iou'] for fold in cv_results['fold_results']],
                'cv_summary': cv_results['cv_summary']
            }
            
            detailed_results[n_folds] = cv_results
            
            print(f"\n{n_folds}-fold CV Results:")
            print(f"  IoU: {mean_iou:.4f} ± {std_iou:.4f}")
            print(f"  F1:  {mean_f1:.4f} ± {std_f1:.4f}")
        
        # Store all results
        self.results = {
            'fold_results': fold_results,
            'detailed_results': detailed_results,
            'config': self.base_config,
            'fold_options_tested': list(fold_results.keys())
        }
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print summary of fold variation results."""
        print(f"\n{'='*60}")
        print("FOLD VARIATION SUMMARY")
        print(f"{'='*60}")
        
        if not self.results['fold_results']:
            print("No results to summarize!")
            return
        
        # Sort by mean IoU
        sorted_folds = sorted(
            self.results['fold_results'].items(),
            key=lambda x: x[1]['mean_iou'],
            reverse=True
        )
        
        print("Results ranked by IoU performance:")
        print("-" * 50)
        
        for i, (n_folds, results) in enumerate(sorted_folds):
            marker = "🏆" if i == 0 else f"{i+1:2d}."
            print(f"{marker} {n_folds:2d} folds: IoU = {results['mean_iou']:.4f} ± {results['std_iou']:.4f}")
        
        # Best configuration
        best_folds, best_results = sorted_folds[0]
        print(f"\n✅ RECOMMENDED: {best_folds} folds")
        print(f"   - Best mean IoU: {best_results['mean_iou']:.4f}")
        print(f"   - Standard deviation: {best_results['std_iou']:.4f}")
        print(f"   - Stability score: {1/best_results['std_iou']:.2f}")
        
        # Statistical significance check
        if len(sorted_folds) > 1:
            second_best = sorted_folds[1]
            improvement = best_results['mean_iou'] - second_best[1]['mean_iou']
            combined_std = np.sqrt(best_results['std_iou']**2 + second_best[1]['std_iou']**2)
            
            if improvement > combined_std:
                print(f"   - Significantly better than {second_best[0]} folds (Δ={improvement:.4f})")
            else:
                print(f"   - Marginal improvement over {second_best[0]} folds (Δ={improvement:.4f})")
    
    def plot_results(self, save_dir: str = None):
        """Create visualization of fold variation results."""
        if not self.results['fold_results']:
            print("No results to plot!")
            return
        
        # Prepare data
        folds = sorted(self.results['fold_results'].keys())
        means = [self.results['fold_results'][f]['mean_iou'] for f in folds]
        stds = [self.results['fold_results'][f]['std_iou'] for f in folds]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Main plot: IoU vs Number of Folds
        plt.subplot(2, 2, 1)
        plt.errorbar(folds, means, yerr=stds, marker='o', linewidth=2, 
                    markersize=8, capsize=5, capthick=2)
        plt.xlabel('Number of CV Folds')
        plt.ylabel('IoU Score')
        plt.title('CV Performance vs Number of Folds')
        plt.grid(True, alpha=0.3)
        
        # Highlight best
        best_idx = np.argmax(means)
        plt.scatter(folds[best_idx], means[best_idx], color='red', s=150, 
                   zorder=5, marker='*', label='Best')
        
        # Add value labels
        for f, m, s in zip(folds, means, stds):
            plt.annotate(f'{m:.3f}', (f, m + s + 0.002), ha='center', fontsize=9)
        
        plt.legend()
        
        # Stability plot: Standard deviation
        plt.subplot(2, 2, 2)
        plt.bar(folds, stds, alpha=0.7, color='orange')
        plt.xlabel('Number of CV Folds')
        plt.ylabel('IoU Standard Deviation')
        plt.title('CV Stability (Lower = More Stable)')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for f, s in zip(folds, stds):
            plt.text(f, s + 0.001, f'{s:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Box plot: Distribution of fold scores
        plt.subplot(2, 2, 3)
        fold_scores_data = [self.results['fold_results'][f]['fold_scores'] for f in folds]
        bp = plt.boxplot(fold_scores_data, labels=[str(f) for f in folds], patch_artist=True)
        
        # Color the best one
        best_idx = np.argmax(means)
        bp['boxes'][best_idx].set_facecolor('lightcoral')
        
        plt.xlabel('Number of CV Folds')
        plt.ylabel('IoU Score')
        plt.title('Distribution of Individual Fold Scores')
        plt.grid(axis='y', alpha=0.3)
        
        # Trade-off plot: Performance vs Stability
        plt.subplot(2, 2, 4)
        plt.scatter(stds, means, s=100, alpha=0.7)
        
        for i, f in enumerate(folds):
            plt.annotate(f'{f} folds', (stds[i], means[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.xlabel('Standard Deviation (Stability)')
        plt.ylabel('Mean IoU (Performance)')
        plt.title('Performance vs Stability Trade-off')
        plt.grid(True, alpha=0.3)
        
        # Add optimal region
        if len(means) > 1:
            max_mean = max(means)
            min_std = min(stds)
            plt.axhline(y=max_mean * 0.99, color='red', linestyle='--', alpha=0.5, 
                       label='99% of best performance')
            plt.axvline(x=min_std * 1.1, color='green', linestyle='--', alpha=0.5,
                       label='110% of best stability')
            plt.legend()
        
        plt.tight_layout()
        
        # Save if directory provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/fold_variation_results.png", dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_dir}/fold_variation_results.png")
        
        plt.show()
    
    def save_results(self, save_dir: str):
        """Save experiment results."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save complete results
        torch.save(self.results, f"{save_dir}/fold_variation_results.pth")
        
        # Save summary
        summary = {
            'experiment_type': 'fold_variation',
            'timestamp': datetime.now().isoformat(),
            'config': self.base_config,
            'fold_options_tested': self.results['fold_options_tested'],
            'best_configuration': None
        }
        
        if self.results['fold_results']:
            best_folds = max(self.results['fold_results'].items(), 
                           key=lambda x: x[1]['mean_iou'])
            summary['best_configuration'] = {
                'n_folds': best_folds[0],
                'mean_iou': best_folds[1]['mean_iou'],
                'std_iou': best_folds[1]['std_iou']
            }
        
        import json
        with open(f"{save_dir}/fold_variation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to {save_dir}/")


def main():
    """Main function to run fold variation experiment."""
    
    # Configuration
    base_config = {
        'name': 'Fold Variation Test',
        'backbone': 'resnet34',
        'use_attention': False,
        'batch_size': 4,
        'num_epochs': 1,  # Reduced for faster testing
        'img_size': (128, 128),
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'pretrained': True,
        'early_stopping_patience': 5,
        'early_stopping_min_delta': 0.001,
        'verbose': False,  # Keep output clean during CV
        'save_plots': False,
        
        # Data configuration
        'data_dir': 'manual_labels',
        'image_type': 'W',
        'random_state': 42,
        'augmentations_per_image': 3,  # Fixed for this experiment
        
        # Loss configuration
        'loss_fn': 'bce'
    }
    
    # Fold options to test
    fold_options = [2, 3, 4, 5, 6, 7]
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"experiments/fold_variation_{timestamp}"
    
    print("="*60)
    print("CV FOLD VARIATION EXPERIMENT")
    print("="*60)
    print(f"Testing fold options: {fold_options}")
    print(f"Results will be saved to: {save_dir}")
    print("="*60)
    
    # Run experiment
    experiment = FoldVariationExperiment(base_config, fold_options)
    results = experiment.run_fold_comparison()
    
    # Save results
    experiment.save_results(save_dir)
    
    # Create plots
    experiment.plot_results(save_dir)
    
    print(f"\n{'='*60}")
    print("FOLD VARIATION EXPERIMENT COMPLETE!")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    # Set random seeds for reproducibility
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Run the experiment
    results = main()
