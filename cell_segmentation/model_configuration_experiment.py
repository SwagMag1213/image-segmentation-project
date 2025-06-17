"""
Model Configuration Experiment using Modular Framework
Tests different model configurations (backbone, attention, batch size, image size)
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Tuple
from itertools import product

# Import from modular framework
from cross_validation import ModelComparator
from advanced_models import UNetWithBackbone


class ModelConfigurationExperiment:
    """Test different model configurations using the modular framework."""
    
    def __init__(self, base_config: Dict, configuration_options: Dict):
        """
        Args:
            base_config: Base configuration for models
            configuration_options: Dictionary of options to vary
                e.g., {
                    'backbone': ['resnet34', 'resnet50'],
                    'use_attention': [True, False],
                    'batch_size': [2, 4],
                    'img_size': [(128, 128), (256, 256)]
                }
        """
        self.base_config = base_config
        self.configuration_options = configuration_options
        self.results = {}
        
        # Generate all configuration combinations
        self.model_configs = self._generate_model_configs()
        
    def _generate_model_configs(self) -> List[Dict]:
        """Generate all combinations of model configurations."""
        print("Generating model configuration combinations...")
        
        # Get all option names and values
        option_names = list(self.configuration_options.keys())
        option_values = list(self.configuration_options.values())
        
        # Generate all combinations
        configurations = []
        for combination in product(*option_values):
            config = self.base_config.copy()
            
            # Apply this combination
            config_params = {}
            for i, option_name in enumerate(option_names):
                config_params[option_name] = combination[i]
                config[option_name] = combination[i]
            
            # Create descriptive name
            name_parts = []
            for option_name, value in config_params.items():
                if option_name == 'use_attention':
                    name_parts.append(f"att_{value}")
                elif option_name == 'img_size':
                    name_parts.append(f"img_{value[0]}x{value[1]}")
                elif option_name == 'batch_size':
                    name_parts.append(f"bs_{value}")
                else:
                    name_parts.append(f"{option_name}_{value}")
            
            config['name'] = "_".join(name_parts)
            config['config_params'] = config_params
            configurations.append(config)
        
        print(f"Generated {len(configurations)} model configurations:")
        for config in configurations:
            print(f"  - {config['name']}")
        
        return configurations
    
    def run_model_comparison(self) -> Dict:
        """
        Test different model configurations using ModelComparator.
        """
        print(f"{'='*60}")
        print("MODEL CONFIGURATION EXPERIMENT")
        print(f"{'='*60}")
        print(f"Testing {len(self.model_configs)} configurations:")
        
        # Create model configurations for ModelComparator
        # Each configuration uses the same model class but different parameters
        model_config_tuples = [(UNetWithBackbone, config) for config in self.model_configs]
        
        # Use ModelComparator for fair comparison
        comparator = ModelComparator(
            data_dir=self.base_config['data_dir'],
            image_type=self.base_config['image_type'],
            test_size=self.base_config.get('test_size', 0.2),
            n_splits=self.base_config.get('n_splits', 5),
            random_state=self.base_config.get('random_state', 42),
            augmentations_per_image=self.base_config.get('augmentations_per_image', 3),
            verbose=True
        )
        
        print(f"\n{'='*50}")
        print("PHASE 1: CROSS-VALIDATION COMPARISON")
        print(f"{'='*50}")
        
        # Run cross-validation comparison
        cv_results = comparator.run_cv_comparison(model_config_tuples)
        
        print(f"\n{'='*50}")
        print("PHASE 2: GENERALIZATION EVALUATION")
        print(f"{'='*50}")
        
        # Run generalization evaluation
        gen_results = comparator.evaluate_generalization(model_config_tuples)
        
        # Store results
        self.results = {
            'cv_results': cv_results,
            'generalization_results': gen_results,
            'model_configs': self.model_configs,
            'configuration_options': self.configuration_options
        }
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print summary of model configuration results."""
        print(f"\n{'='*60}")
        print("MODEL CONFIGURATION SUMMARY")
        print(f"{'='*60}")
        
        cv_summary = self.results['cv_results']['comparison_summary']
        gen_results = self.results['generalization_results']
        
        if not cv_summary:
            print("No results to summarize!")
            return
        
        # Best by CV
        best_cv = max(cv_summary.items(), key=lambda x: x[1]['iou_mean'])
        print(f"\n✅ BEST BY CROSS-VALIDATION:")
        print(f"   Configuration: {best_cv[0]}")
        print(f"   CV IoU: {best_cv[1]['iou_mean']:.4f} ± {best_cv[1]['iou_std']:.4f}")
        
        # Best by Test
        best_test = max(gen_results.items(), key=lambda x: x[1]['test_metrics']['iou'])
        print(f"\n🎯 BEST BY TEST PERFORMANCE:")
        print(f"   Configuration: {best_test[0]}")
        print(f"   Test IoU: {best_test[1]['test_metrics']['iou']:.4f}")
        
        # Check consistency
        if best_cv[0] == best_test[0]:
            print(f"\n🏆 CONSISTENT WINNER: {best_cv[0]}")
            print("   Cross-validation successfully identified the best configuration!")
        else:
            print(f"\n⚠️  DIFFERENT WINNERS:")
            print(f"   CV Best: {best_cv[0]}")
            print(f"   Test Best: {best_test[0]}")
        
        # Calculate correlation
        cv_ious = [cv_summary[name]['iou_mean'] for name in cv_summary.keys()]
        test_ious = [gen_results[name]['test_metrics']['iou'] for name in cv_summary.keys()]
        correlation = np.corrcoef(cv_ious, test_ious)[0, 1]
        
        print(f"\n📊 CV-TEST CORRELATION: r = {correlation:.3f}")
        if correlation > 0.8:
            print("   ✅ Strong correlation - CV is reliable for model selection")
        elif correlation > 0.5:
            print("   ⚠️  Moderate correlation - CV provides reasonable guidance")
        else:
            print("   ❌ Weak correlation - Consider larger validation sets")
        
        # Component analysis
        self._analyze_configuration_effects()
    
    def _analyze_configuration_effects(self):
        """Analyze the effect of individual configuration components."""
        print(f"\n{'='*50}")
        print("COMPONENT ANALYSIS")
        print(f"{'='*50}")
        
        gen_results = self.results['generalization_results']
        
        # Group results by configuration components
        component_effects = {}
        
        for option_name in self.configuration_options.keys():
            component_effects[option_name] = {}
            
            # Get unique values for this option
            unique_values = self.configuration_options[option_name]
            
            for value in unique_values:
                # Find all configurations with this value
                matching_configs = []
                for config in self.model_configs:
                    if config['config_params'][option_name] == value:
                        config_name = config['name']
                        if config_name in gen_results:
                            matching_configs.append(gen_results[config_name]['test_metrics']['iou'])
                
                if matching_configs:
                    component_effects[option_name][value] = {
                        'mean_iou': np.mean(matching_configs),
                        'std_iou': np.std(matching_configs),
                        'count': len(matching_configs)
                    }
        
        # Print component effects
        for option_name, effects in component_effects.items():
            print(f"\n{option_name.upper()} EFFECTS:")
            sorted_effects = sorted(effects.items(), key=lambda x: x[1]['mean_iou'], reverse=True)
            
            for i, (value, stats) in enumerate(sorted_effects):
                marker = "🏆" if i == 0 else f"{i+1}."
                print(f"  {marker} {value}: {stats['mean_iou']:.4f} ± {stats['std_iou']:.4f} "
                      f"({stats['count']} configs)")
    
    def plot_results(self, save_dir: str = None):
        """Create comprehensive visualization of model configuration results."""
        if not self.results:
            print("No results to plot!")
            return
        
        cv_summary = self.results['cv_results']['comparison_summary']
        gen_results = self.results['generalization_results']
        
        # Set up the plot
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(20, 16))
        
        # Prepare data
        config_names = list(cv_summary.keys())
        cv_ious = [cv_summary[name]['iou_mean'] for name in config_names]
        cv_stds = [cv_summary[name]['iou_std'] for name in config_names]
        test_ious = [gen_results[name]['test_metrics']['iou'] for name in config_names]
        
        # Plot 1: CV Performance (sorted)
        plt.subplot(3, 3, 1)
        cv_sorted_idx = np.argsort(cv_ious)[::-1]
        sorted_names = [config_names[i] for i in cv_sorted_idx]
        sorted_cv_ious = [cv_ious[i] for i in cv_sorted_idx]
        sorted_cv_stds = [cv_stds[i] for i in cv_sorted_idx]
        
        bars = plt.bar(range(len(sorted_names)), sorted_cv_ious, 
                      yerr=sorted_cv_stds, capsize=5, alpha=0.7)
        plt.xticks(range(len(sorted_names)), [name.replace('_', '\n') for name in sorted_names], 
                  rotation=45, ha='right', fontsize=8)
        plt.ylabel('IoU')
        plt.title('CV Performance (Ranked)', fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # Plot 2: Test Performance (sorted)
        plt.subplot(3, 3, 2)
        test_sorted_idx = np.argsort(test_ious)[::-1]
        test_sorted_names = [config_names[i] for i in test_sorted_idx]
        test_sorted_ious = [test_ious[i] for i in test_sorted_idx]
        
        bars = plt.bar(range(len(test_sorted_names)), test_sorted_ious, alpha=0.7, color='orange')
        plt.xticks(range(len(test_sorted_names)), [name.replace('_', '\n') for name in test_sorted_names], 
                  rotation=45, ha='right', fontsize=8)
        plt.ylabel('IoU')
        plt.title('Test Performance (Ranked)', fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # Plot 3: CV vs Test Correlation
        plt.subplot(3, 3, 3)
        plt.scatter(cv_ious, test_ious, alpha=0.7, s=100)
        
        # Add correlation line
        z = np.polyfit(cv_ious, test_ious, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(cv_ious), max(cv_ious), 100)
        plt.plot(x_line, p(x_line), "r--", alpha=0.8)
        
        correlation = np.corrcoef(cv_ious, test_ious)[0, 1]
        plt.xlabel('CV IoU')
        plt.ylabel('Test IoU')
        plt.title(f'CV vs Test Correlation\n(r = {correlation:.3f})', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Component effect plots (dynamic based on configuration options)
        plot_idx = 4
        for option_name in self.configuration_options.keys():
            if plot_idx > 9:  # Limit to available subplot positions
                break
                
            plt.subplot(3, 3, plot_idx)
            
            # Group results by this option
            option_results = {}
            for config in self.model_configs:
                option_value = config['config_params'][option_name]
                config_name = config['name']
                
                if config_name in gen_results:
                    if option_value not in option_results:
                        option_results[option_value] = []
                    option_results[option_value].append(gen_results[config_name]['test_metrics']['iou'])
            
            # Create bar plot
            option_labels = []
            option_means = []
            option_stds = []
            
            for value, ious in option_results.items():
                option_labels.append(str(value))
                option_means.append(np.mean(ious))
                option_stds.append(np.std(ious))
            
            bars = plt.bar(option_labels, option_means, yerr=option_stds, 
                          capsize=5, alpha=0.7)
            plt.ylabel('Test IoU')
            plt.title(f'Effect of {option_name.replace("_", " ").title()}', fontweight='bold')
            plt.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, mean in zip(bars, option_means):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(option_stds)*0.1,
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
            
            plot_idx += 1
        
        # Configuration interaction heatmap (if we have exactly 2 main factors)
        main_factors = ['backbone', 'use_attention', 'batch_size', 'img_size']
        available_factors = [f for f in main_factors if f in self.configuration_options]
        
        if len(available_factors) >= 2 and plot_idx <= 9:
            plt.subplot(3, 3, plot_idx)
            
            factor1, factor2 = available_factors[:2]
            
            # Create interaction matrix
            factor1_values = self.configuration_options[factor1]
            factor2_values = self.configuration_options[factor2]
            
            interaction_matrix = np.zeros((len(factor1_values), len(factor2_values)))
            
            for i, val1 in enumerate(factor1_values):
                for j, val2 in enumerate(factor2_values):
                    # Find configs with this combination
                    matching_ious = []
                    for config in self.model_configs:
                        if (config['config_params'][factor1] == val1 and 
                            config['config_params'][factor2] == val2):
                            config_name = config['name']
                            if config_name in gen_results:
                                matching_ious.append(gen_results[config_name]['test_metrics']['iou'])
                    
                    if matching_ious:
                        interaction_matrix[i, j] = np.mean(matching_ious)
                    else:
                        interaction_matrix[i, j] = np.nan
            
            # Create heatmap
            sns.heatmap(interaction_matrix, 
                       xticklabels=[str(v) for v in factor2_values],
                       yticklabels=[str(v) for v in factor1_values],
                       annot=True, fmt='.3f', cmap='viridis')
            plt.xlabel(factor2.replace('_', ' ').title())
            plt.ylabel(factor1.replace('_', ' ').title())
            plt.title(f'{factor1} × {factor2} Interaction', fontweight='bold')
        
        plt.tight_layout()
        
        # Save if directory provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/model_configuration_results.png", dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_dir}/model_configuration_results.png")
        
        plt.show()
    
    def save_results(self, save_dir: str):
        """Save experiment results."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save complete results
        torch.save(self.results, f"{save_dir}/model_configuration_results.pth")
        
        # Save summary
        summary = {
            'experiment_type': 'model_configuration',
            'timestamp': datetime.now().isoformat(),
            'base_config': self.base_config,
            'configuration_options': self.configuration_options,
            'total_configurations_tested': len(self.model_configs),
            'best_cv_config': None,
            'best_test_config': None
        }
        
        if self.results:
            cv_summary = self.results['cv_results']['comparison_summary']
            gen_results = self.results['generalization_results']
            
            if cv_summary:
                best_cv = max(cv_summary.items(), key=lambda x: x[1]['iou_mean'])
                summary['best_cv_config'] = {
                    'name': best_cv[0],
                    'mean_iou': best_cv[1]['iou_mean'],
                    'std_iou': best_cv[1]['iou_std']
                }
            
            if gen_results:
                best_test = max(gen_results.items(), key=lambda x: x[1]['test_metrics']['iou'])
                summary['best_test_config'] = {
                    'name': best_test[0],
                    'test_iou': best_test[1]['test_metrics']['iou'],
                    'test_f1': best_test[1]['test_metrics']['f1']
                }
        
        import json
        with open(f"{save_dir}/model_configuration_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Results saved to {save_dir}/")


def main():
    """Main function to run model configuration experiment."""
    
    # Base configuration (fixed parameters)
    base_config = {
        'name': 'Model Configuration Test',
        'num_epochs': 1,  # Reduced for faster testing
        'learning_rate': 1e-3,
        'weight_decay': 1e-8,
        'pretrained': True,
        'early_stopping_patience': 5,
        'early_stopping_min_delta': 0.001,
        'verbose': False,  # Keep output clean during CV
        'save_plots': False,
        
        # Data configuration
        'data_dir': 'manual_labels',
        'image_type': 'W',
        'test_size': 0.2,
        'n_splits': 5,
        'random_state': 42,
        'augmentations_per_image': 3,
        
        # Loss configuration (use your best loss function)
        'loss_fn': 'bce'
    }
    
    # Configuration options to test
    configuration_options = {
        'backbone': ['resnet34', 'resnet50'],
        'use_attention': [True, False],
        'batch_size': [2, 4],
        'img_size': [(128, 128), (256, 256), (512, 512)]  # Removed (512, 512) for faster testing
    }
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"experiments/model_configuration_{timestamp}"
    
    print("="*60)
    print("MODEL CONFIGURATION EXPERIMENT")
    print("="*60)
    print(f"Testing combinations of:")
    for option, values in configuration_options.items():
        print(f"  {option}: {values}")
    print(f"Total configurations: {np.prod([len(v) for v in configuration_options.values()])}")
    print(f"Results will be saved to: {save_dir}")
    print("="*60)
    
    # Run experiment
    experiment = ModelConfigurationExperiment(base_config, configuration_options)
    results = experiment.run_model_comparison()
    
    # Save results
    experiment.save_results(save_dir)
    
    # Create plots
    experiment.plot_results(save_dir)
    
    print(f"\n{'='*60}")
    print("MODEL CONFIGURATION EXPERIMENT COMPLETE!")
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
