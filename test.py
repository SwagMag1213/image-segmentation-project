"""
Simple test script for cross-validation system
Tests CV on a single model to verify everything works
"""

import torch
import numpy as np
from cross_validation import CrossValidator
from advanced_models import UNetWithBackbone  # or whatever your model class is called

def test_single_model_cv():
    """Test cross-validation on a single model."""
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Define a simple test configuration
    config = {
        'name': 'fold_augmentation_test',
        'model_type': 'unet',
        'image_type': 'W',  # or 'B'
        'backbone': 'resnet34',
        'use_attention': True,
        'batch_size': 4,
        'img_size': (128, 128),
        'num_epochs': 15,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'pretrained': True,
        'seed': 42,
        'loss_fn': 'focal',
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
    }
    
    print("="*50)
    print("TESTING CROSS-VALIDATION SYSTEM")
    print("="*50)
    print(f"Model: {config['backbone']}")
    print(f"Epochs: {config['num_epochs']} (reduced for testing)")
    print(f"Loss: {config['loss_fn']}")
    print("="*50)
    
    try:
        # Initialize cross-validator
        cv = CrossValidator(
            data_dir="manual_labels",  # Update this path if needed
            image_type='W',            # Change to 'B' if you want fluorescent images
            n_splits=3,                # Reduced for faster testing
            augmentations_per_image=2, # Reduced for faster testing
            verbose=True
        )
        
        # Run cross-validation
        results = cv.cross_validate_single_model(UNetWithBackbone, config)
        
        # Print results
        print("\n" + "="*50)
        print("CROSS-VALIDATION COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        cv_summary = results['cv_summary']
        print(f"Mean IoU: {cv_summary['iou_mean']:.4f} ± {cv_summary['iou_std']:.4f}")
        print(f"Mean F1:  {cv_summary['f1_mean']:.4f} ± {cv_summary['f1_std']:.4f}")
        print(f"Mean Precision: {cv_summary['precision_mean']:.4f} ± {cv_summary['precision_std']:.4f}")
        print(f"Mean Recall: {cv_summary['recall_mean']:.4f} ± {cv_summary['recall_std']:.4f}")
        
        print(f"\nFold Results:")
        for i, fold_result in enumerate(results['fold_results']):
            print(f"  Fold {i+1}: IoU = {fold_result['iou']:.4f}, "
                  f"Training Time = {fold_result['training_time']:.1f}s")
        
        print("\n✅ Cross-validation system is working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during cross-validation: {str(e)}")
        print("\nCommon issues to check:")
        print("1. Make sure 'manual_labels' directory exists and contains data")
        print("2. Check that you have the required model class imported")
        print("3. Verify all dependencies are installed")
        print("4. Check file paths in load_original_data()")
        return False

if __name__ == "__main__":
    # Run the test
    success = test_single_model_cv()
    
    if success:
        print("\n🎉 Your modular CV system is ready to use!")
        print("\nNext steps:")
        print("- Try different loss functions")
        print("- Test model comparison with multiple models")
        print("- Run the full loss function comparison experiment")
    else:
        print("\n🔧 Please fix the issues above and try again.")