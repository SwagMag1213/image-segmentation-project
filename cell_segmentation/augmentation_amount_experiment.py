"""
Augmentation Amount Experiment using Modular Framework
Tests different amounts of data augmentation to find optimal configuration
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


class AugmentationAmountExperiment:
    """Test different amounts of data augmentation using the modular framework."""
    
    def __init__(self, base_config: Dict, augmentation_amounts: List[int] = [0, 1, 2, 3, 5, 10]):
        """
        Args:
            base_config: Base configuration for models
            augmentation_amounts: Different numbers of augmentations per image to test
        """
        self.base_config = base_config
        self.augmentation_amounts = augmentation_amounts
        self.results = {}
        
    def run_augmentation_comparison(self) -> Dict:
        """
        Test different amounts of data augmentation using the same model configuration.
        Uses CrossValidator with different augmentations_per_image values.
        """
        print(f"{'='*60}")
        print("AUGMENTATION AMOUNT EXPERIMENT")
        print(f"{'='*60}")
        print(f"Model: {self.base_config['backbone']} UNet")
        print(f"Testing augmentation amounts: {self.augmentation_amounts}")
        print(f"Fixed CV folds: {self.base_config.get('n_splits', 5)}")
        
        # Results storage
        aug_results = {}
        detailed_results = {}
        
        for aug_amount in self.augmentation_amounts:
            print(f"\n{'='*40}")
            print(f"TESTING {aug_amount} AUGMENTATIONS PER IMAGE")
            print(f"{'='*40}")
            
            # Create modified config for this augmentation amount
            current_config = self.base_config.copy()
            current_config['name'] = f"{aug_amount} Augmentations"
            
            # Create CrossValidator with this augmentation configuration
            cv = CrossValidator(
                data_dir=self.base_config['data_dir'],
                image_type=self.base_config['image_type'],
                n_splits=self.base_config.get('n_splits', 5),
                random_state=self.base_config['random_state'],
                augmentations_per_image=aug_amount,  # This is the key parameter we're testing
                verbose=True
            )
            
            # Calculate total training samples (original + augmented)
            original_samples = len(cv.image_paths)
            total_samples_per_fold = original_samples * (1 + aug_amount) * (cv.n_splits - 1) / cv.n_splits
            
            print(f"Original samples: {original_samples}")
            print(f"Augmentation multiplier: {1 + aug_amount}x")
            print(f"Approximate training samples per fold: {total_samples_per_fold:.0f}")
            
            # Run cross-validation for this augmentation configuration
            cv_results = cv.cross_validate_single_model(
                model_class=UNetWithBackbone,
                config=current_config
            )
            
            # Extract key metrics
            mean_iou = cv_results['cv_summary']['iou_mean']
            std_iou = cv_results['cv_summary']['iou_std']
            mean_f1 = cv_results['cv_summary']['f1_mean']
            std_f1 = cv_results['cv_summary']['f1_std']
            
            # Calculate training efficiency metrics
            efficiency_iou = mean_iou / (1 + aug_amount)  # IoU per data multiplier
            
            # Store results
            aug_results[aug_amount] = {
                'mean_iou': mean_iou,
                'std_iou': std_iou,
                'mean_f1': mean_f1,
                'std_f1': std_f1,
                'fold_scores': [fold['iou'] for fold in cv_results['fold_results']],
                'data_multiplier': 1 + aug_amount,
                'efficiency_iou': efficiency_iou,
                'total_samples_per_fold': int(total_samples_per_fold),
                'cv_summary': cv_results['cv_summary']
            }
            
            detailed_results[aug_amount] = cv_results
            
            print(f"\n{aug_amount} augmentations Results:")
            print(f"  IoU: {mean_iou:.4f} ± {std_iou:.4f}")
            print(f"  F1:  {mean_f1:.4f} ± {std_f1:.4f}")
            print(f"  Efficiency: {efficiency_iou:.4f} IoU per data multiplier")
        
        # Store all results
        self.results = {
            'augmentation_results': aug_results,
            'detailed_results': detailed_results,
            'config': self.base_config,
            'augmentation_amounts_tested': list(aug_results.keys())
        }
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print summary of augmentation amount results."""
        print(f"\n{'='*60}")
        print("AUGMENTATION AMOUNT SUMMARY")
        print(f"{'='*60}")
        
        if not self.results['augmentation_results']:
            print("No results to summarize!")
            return
        
        # Sort by mean IoU
        sorted_augs = sorted(
            self.results['augmentation_results'].items(),
            key=lambda x: x[1]['mean_iou'],
            reverse=True
        )
        
        print("Results ranked by IoU performance:")
        print("-" * 65)
        print("Rank | Aug/Img | IoU ± Std    | Data Mult | Efficiency")
        print("-" * 65)
        
        for i, (aug_amount, results) in enumerate(sorted_augs):
            marker = "🏆" if i == 0 else f"{i+1:2d}."
            print(f"{marker} | {aug_amount:7d} | {results['mean_iou']:.4f} ± {results['std_iou']:.4f} | "
                  f"{results['data_multiplier']:8.0f}x | {results['efficiency_iou']:.4f}")
        
        # Best configurations
        best_aug, best_results = sorted_augs[0]
        print(f"\n✅ BEST PERFORMANCE: {best_aug} augmentations per image")
        print(f"   - Best mean IoU: {best_results['mean_iou']:.4f}")
        print(f"   - Standard deviation: {best_results['std_iou']:.4f}")
        print(f"   - Data multiplier: {best_results['data_multiplier']}x")
        
        # Best efficiency
        best_efficiency = max(self.results['augmentation_results'].items(), 
                            key=lambda x: x[1]['efficiency_iou'])
        
        if best_efficiency[0] != best_aug:
            print(f"\n⚡ BEST EFFICIENCY: {best_efficiency[0]} augmentations per image")
            print(f"   - Efficiency score: {best_efficiency[1]['efficiency_iou']:.4f}")
            print(f"   - IoU: {best_efficiency[1]['mean_iou']:.4f}")
            print(f"   - Data multiplier: {best_efficiency[1]['data_multiplier']}x")
        
        # Diminishing returns analysis
        if len(sorted_augs) > 1:
            print(f"\n📊 AUGMENTATION ANALYSIS:")
            
            # Find baseline (0 augmentations) or minimum
            baseline_aug = min(self.results['augmentation_results'].keys())
            baseline_iou = self.results['augmentation_results'][baseline_aug]['mean_iou']
            
            for aug_amount, results in sorted(self.results['augmentation_results'].items()):
                if aug_amount == baseline_aug:
                    continue
                
                improvement = results['mean_iou'] - baseline_iou
                cost = aug_amount  # Additional augmentations
                roi = improvement / cost if cost > 0 else 0
                
                print(f"   - {aug_amount} aug: +{improvement:.4f} IoU for {cost} extra aug (ROI: {roi:.4f})")
    
    def plot_results(self, save_dir: str = None):
        """Create visualization of augmentation amount results."""
        if not self.results['augmentation_results']:
            print("No results to plot!")
            return
        
        # Prepare data
        aug_amounts = sorted(self.results['augmentation_results'].keys())
        means = [self.results['augmentation_results'][a]['mean_iou'] for a in aug_amounts]
        stds = [self.results['augmentation_results'][a]['std_iou'] for a in aug_amounts]
        data_multipliers = [self.results['augmentation_results'][a]['data_multiplier'] for a in aug_amounts]
        efficiencies = [self.results['augmentation_results'][a]['efficiency_iou'] for a in aug_amounts]
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot 1: IoU vs Augmentation Amount
        plt.subplot(2, 3, 1)
        plt.errorbar(aug_amounts, means, yerr=stds, marker='o', linewidth=2, 
                    markersize=8, capsize=5, capthick=2, color='blue')
        plt.xlabel('Augmentations per Image')
        plt.ylabel('IoU Score')
        plt.title('Performance vs Augmentation Amount')
        plt.grid(True, alpha=0.3)
        
        # Highlight best
        best_idx = np.argmax(means)
        plt.scatter(aug_amounts[best_idx], means[best_idx], color='red', s=150, 
                   zorder=5, marker='*', label='Best Performance')
        
        # Add value labels
        for a, m, s in zip(aug_amounts, means, stds):
            plt.annotate(f'{m:.3f}', (a, m + s + 0.002), ha='center', fontsize=9)
        
        plt.legend()
        
        # Plot 2: Efficiency Analysis
        plt.subplot(2, 3, 2)
        plt.plot(aug_amounts, efficiencies, marker='s', linewidth=2, 
                markersize=8, color='green')
        plt.xlabel('Augmentations per Image')
        plt.ylabel('IoU per Data Multiplier')
        plt.title('Training Efficiency')
        plt.grid(True, alpha=0.3)
        
        # Highlight best efficiency
        best_eff_idx = np.argmax(efficiencies)
        plt.scatter(aug_amounts[best_eff_idx], efficiencies[best_eff_idx], 
                   color='red', s=150, zorder=5, marker='*', label='Best Efficiency')
        plt.legend()
        
        # Plot 3: IoU vs Data Multiplier
        plt.subplot(2, 3, 3)
        plt.scatter(data_multipliers, means, s=100, alpha=0.7, color='purple')
        
        for i, a in enumerate(aug_amounts):
            plt.annotate(f'{a} aug', (data_multipliers[i], means[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.xlabel('Data Multiplier')
        plt.ylabel('IoU Score')
        plt.title('Performance vs Dataset Size')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Box plot of fold scores
        plt.subplot(2, 3, 4)
        fold_scores_data = [self.results['augmentation_results'][a]['fold_scores'] for a in aug_amounts]
        bp = plt.boxplot(fold_scores_data, labels=[str(a) for a in aug_amounts], patch_artist=True)
        
        # Color the best one
        best_idx = np.argmax(means)
        bp['boxes'][best_idx].set_facecolor('lightcoral')
        
        plt.xlabel('Augmentations per Image')
        plt.ylabel('IoU Score')
        plt.title('Distribution of Fold Scores')
        plt.grid(axis='y', alpha=0.3)
        
        # Plot 5: Improvement over baseline
        plt.subplot(2, 3, 5)
        if aug_amounts:
            baseline_iou = means[0]  # First augmentation amount (usually 0)
            improvements = [m - baseline_iou for m in means]
            
            plt.bar(aug_amounts, improvements, alpha=0.7, color='orange')
            plt.xlabel('Augmentations per Image')
            plt.ylabel('IoU Improvement over Baseline')
            plt.title('Augmentation Benefit')
            plt.grid(axis='y', alpha=0.3)
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            # Add value labels
            for a, imp in zip(aug_amounts, improvements):
                if imp >= 0:
                    plt.text(a, imp + 0.001, f'+{imp:.3f}', ha='center', va='bottom', fontsize=9)
                else:
                    plt.text(a, imp - 0.001, f'{imp:.3f}', ha='center', va='top', fontsize=9)
        
        # Plot 6: Training cost analysis
        plt.subplot(2, 3, 6)
        training_costs = [a + 1 for a in aug_amounts]  # Relative training time
        
        plt.scatter(training_costs, means, s=100, alpha=0.7, color='brown')
        
        for i, a in enumerate(aug_amounts):
            plt.annotate(f'{a} aug', (training_costs[i], means[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.xlabel('Relative Training Cost')
        plt.ylabel('IoU Score')
        plt.title('Performance vs Training Cost')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if directory provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/augmentation_amount_results.png", dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_dir}/augmentation_amount_results.png")
        
        plt.show()
    
    def save_results(self, save_dir: str):
        """Save experiment results."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save complete results
        torch.save(self.results, f"{save_dir}/augmentation_amount_results.pth")
        
        # Save summary
        summary = {
            'experiment_type': 'augmentation_amount',
            'timestamp': datetime.now().isoformat(),
            'config': self.base_config,
            'augmentation_amounts_tested': self.results['augmentation_amounts_tested'],
            'best_performance': None,
            'best_efficiency': None
        }
        
        if self.results['augmentation_results']:
            # Best performance
            best_perf = max(self.results['augmentation_results'].items(), 
                          key=lambda x: x[1]['mean_iou'])
            summary['best_performance'] = {
                'augmentations_per_image': best_perf[0],
                'mean_iou': best_perf[1]['mean_iou'],
                'std_iou': best_perf[1]['std_iou'],
                'data_multiplier': best_perf[1]['data_multiplier']
            }
            
            # Best efficiency
            best_eff = max(self.results['augmentation_results'].items(), 
                         key=lambda x: x[1]['efficiency_iou'])
            summary['best_efficiency'] = {
                'augmentations_per_image': best_eff[0],
                'efficiency_score': best_eff[1]['efficiency_iou'],
                'mean_iou': best_eff[1]['mean_iou'],
                'data_multiplier': best_eff[1]['data_multiplier']
            }
        
        import json
        with open(f"{save_dir}/augmentation_amount_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to {save_dir}/")


def main():
    """Main function to run augmentation amount experiment."""
    
    # Configuration
    base_config = {
        'name': 'Augmentation Amount Test',
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
        'n_splits': 5,  # Fixed for this experiment
        
        # Loss configuration
        'loss_fn': 'bce'
    }
    
    # Augmentation amounts to test
    augmentation_amounts = []
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"experiments/augmentation_amount_{timestamp}"
    
    print("="*60)
    print("AUGMENTATION AMOUNT EXPERIMENT")
    print("="*60)
    print(f"Testing augmentation amounts: {augmentation_amounts}")
    print(f"Results will be saved to: {save_dir}")
    print("="*60)
    
    # Run experiment
    experiment = AugmentationAmountExperiment(base_config, augmentation_amounts)
    results = experiment.run_augmentation_comparison()
    
    # Save results
    experiment.save_results(save_dir)
    
    # Create plots
    experiment.plot_results(save_dir)
    
    print(f"\n{'='*60}")
    print("AUGMENTATION AMOUNT EXPERIMENT COMPLETE!")
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
